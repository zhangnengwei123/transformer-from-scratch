import os

import torch
import torch.distributed as torch_distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from megatron.core import parallel_state
from nn_transformer import MyTransformer

# 配置参数

# 填充空白
sequence_empty_pad = "[PAD]"
# 标记每个输入的开头
sequence_start_pad = "[CLS]"
# 标记每个输入的结束，用于分割语句
sequence_end_pad = "[SEP]"

sequence_max_length = 12

batch_size = 3

# 准备数据
# 先用这个数据作为训练数据，模型训练完成之后，
# 向模型输入这个source_sentence,模型能输出target_sentence 那就很满足。
# 比如输入 你真是个傻逼 可以得到 你真是个小可爱

source_sentences = ["你真是个傻逼", "你咋这么能呢", "我将带头冲锋", "在东南亚打自由搏击", "铃儿响叮当"]
target_sentences = ["你真是个小可爱", "你就是个菜鸡", "我先撤你们上", "这货真牛逼", "今天真开心"]

# 建立词典
# 提取所有字符
all_characters = set()
all_characters.add(sequence_empty_pad)
all_characters.add(sequence_start_pad)
all_characters.add(sequence_end_pad)
for sentence in source_sentences + target_sentences:
    for char in sentence:
        all_characters.add(char)

# 去重并排序
sorted_characters = sorted(all_characters)

vocabulary_size = len(sorted_characters)  # 词汇表大小

# 构建字典
char_to_index = {char: idx for idx, char in enumerate(sorted_characters)}
index_to_char = {idx: char for idx, char in enumerate(sorted_characters)}

pad_index = char_to_index[sequence_empty_pad]
cls_token_id = char_to_index[sequence_start_pad]
seq_token_id = char_to_index[sequence_end_pad]


def tokenize(sequence):
    return [token for token in sequence]


def convert_tokens_to_ids(tokens):
    return [char_to_index[token_item] for token_item in tokens]


def convert_ids_to_tokens(ids):
    return [index_to_char[id_item] for id_item in ids]


def get_tokens_batch():
    ixs = torch.randint(0, len(source_sentences), (batch_size,))
    print(ixs)
    source_tokens_batch = []
    target_tokens_batch = []
    for i in ixs:
        source_tokens = ['[CLS]'] + tokenize(source_sentences[i]) + ['[SEP]']
        if len(source_tokens) < sequence_max_length:
            # 填充（PAD）并创建注意力掩码
            padding_length_source = sequence_max_length - len(source_tokens)
            source_tokens += [sequence_empty_pad] * padding_length_source
        source_tokens_batch.append(source_tokens)

        target_tokens = ['[CLS]'] + tokenize(target_sentences[i]) + ['[SEP]']
        if len(target_tokens) < sequence_max_length:
            # 填充（PAD）并创建注意力掩码
            padding_length_target = sequence_max_length - len(target_tokens)
            target_tokens += [sequence_empty_pad] * padding_length_target
        target_tokens_batch.append(target_tokens)

    return source_tokens_batch, target_tokens_batch


def embedding(tokens_batch):
    return torch.tensor([convert_tokens_to_ids(tokens) for tokens in tokens_batch])


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    r"""
    初始化训练的设置
    设置并行度
    在 Megatron-LM 中，tensor_model_parallel_size 是一个关键的参数，用于指定张量并行的大小。
    通过将模型的张量计算分配到多个 GPU 上，从而使得能够训练更大的模型。
    pipeline_model_parallel_size 是一个关键参数，用于指定流水线并行的大小。
    通过将模型的不同层分配到多个 GPU 上，并以流水线方式处理数据批次，可以显著降低每个 GPU 上的内存占用，并提高训练吞吐量。
    在配置时，需要根据你的硬件资源、模型深度和计算需求来选择合适的 pipeline_model_parallel_size
    """
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch_distributed.init_process_group(world_size=world_size, rank=rank)

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tensor_model_parallel_size,
                                             pipeline_model_parallel_size=pipeline_model_parallel_size)


def model_provider():
    return MyTransformer(i_vocabulary_size=vocabulary_size,
                         t_vocabulary_size=vocabulary_size,
                         src_pad_idx=pad_index,
                         tgt_pad_idx=pad_index,
                         d_model=512,
                         num_heads=8,
                         num_encoder_layers=6,
                         num_decoder_layers=6,
                         dim_feedforward=2048,
                         dropout=0.1
                         )


def save_distributed_checkpoint(checkpoint_path, trained_model, trained_optimizer):
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': trained_optimizer.state_dict(),
    }, checkpoint_path)


def load_distributed_checkpoint(checkpoint_path, trained_model):
    # 加载模型
    checkpoint = torch.load(checkpoint_path)
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    return trained_model


if __name__ == '__main__':
    print("initialize_distributed")
    # 初始化环境
    initialize_distributed()

    # 拿到模型
    model = model_provider()

    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 循环训练

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)  # 忽略填充值0
    # 计算损失
    # loss = criterion(outputs, targets) 输入参数的维度是[11,vocabulary_size] [11]
    # loss = criterion(logits, target_ids_shifted)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 循环训练
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x, y = get_tokens_batch()
        x_embedding = embedding(x).to(device=device)
        y_embedding = embedding(y).to(device=device)

        logits = model(x_embedding, y_embedding)
        logits = logits.contiguous().view(-1, vocabulary_size).to(device)

        # 向后移动一位 再计算损失
        decoder_inputs_embedding = y_embedding[:, 1:]
        decoder_inputs_embedding = F.pad(decoder_inputs_embedding, (0, 1), value=pad_index)
        decoder_inputs_embedding = decoder_inputs_embedding.contiguous().view(-1)

        decoder_inputs_embedding.to(device)
        # 计算损失
        loss = criterion(logits, decoder_inputs_embedding)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Saving the model
    save_distributed_checkpoint(checkpoint_path='/workspace/nn_transformer_ckpt',
                                trained_model=model,
                                trained_optimizer=optimizer)

    # Loading the model
    model = load_distributed_checkpoint(checkpoint_path='/workspace/nn_transformer_ckpt', trained_model=model)
    model.to(device=device)
    print("Successfully load the model.")

    # Using the model
