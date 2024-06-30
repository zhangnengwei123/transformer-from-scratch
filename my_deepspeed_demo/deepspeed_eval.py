import deepspeed
import argparse
import torch
from nn_transformer import MyTransformer
import torch.distributed as torch_distributed

# ---------------------------------transformer模型相关
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


model = MyTransformer(i_vocabulary_size=vocabulary_size,
                      t_vocabulary_size=vocabulary_size,
                      src_pad_idx=pad_index,
                      tgt_pad_idx=pad_index,
                      d_model=512,
                      num_heads=8,
                      num_encoder_layers=6,
                      num_decoder_layers=6,
                      dim_feedforward=2048,
                      dropout=0.1
                      ).cuda()

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser.add_argument('--epoch', type=int, default=1, help='epoch')
parser = deepspeed.add_config_arguments(parser)

cmd_args = parser.parse_args()

model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=model.parameters())

model_engine.load_checkpoint('./checkpoint/ds_transformer')


def translate(model_engine, source_sentence, max_target_length=12):
    # 将源句子标记化并转换为ID
    source_tokens = tokenize(source_sentence)

    source_tokens = ['[CLS]'] + source_tokens + ['[SEP]']
    padding_length_source = max_target_length - len(source_tokens)
    source_tokens = source_tokens + [sequence_empty_pad] * padding_length_source
    print(f"source_tokens:{source_tokens}")

    source_ids = torch.tensor([convert_tokens_to_ids(source_tokens)]).cuda()
    print(f"source_ids:{source_ids.shape}")

    # attention_mask_source = create_pad_mask(source_ids, pad_index)

    # 初始化解码器输入
    decoder_input_ids = torch.tensor([[cls_token_id]]).cuda()
    print(decoder_input_ids.shape)

    # 用于存储生成的目标序列
    generated_ids = []

    for _ in range(max_target_length):
        # 前向传播
        with torch.no_grad():
            logits = model_engine(source_ids, decoder_input_ids)

        # 获取当前时间步的预测结果
        # [1, 21128]
        next_token_logits = logits[:, -1, :]
        # 找出概率最大的哪一个
        next_token_id = next_token_logits.argmax(dim=-1).item()
        print(f"next_token_id : {next_token_id}")

        # 添加到生成的序列中
        generated_ids.append(next_token_id)
        decoder_input_ids.cuda()
        # 更新解码器输入
        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[next_token_id]]).cuda()], dim=-1)
        print(decoder_input_ids)
        # 如果预测到了[SEP]标记，则停止生成
        if next_token_id == seq_token_id:
            break

    # 转换生成的ID为标记
    generated_tokens = convert_ids_to_tokens(generated_ids)
    return generated_tokens


# 主控进程带头做这些动作 只有主进程可以执行这些动作
if torch_distributed.get_rank() == 0:
    # 测试
    source_sentence = "在东南亚打自由搏击"
    translation = translate(model_engine, source_sentence)
    print(f"翻译: {translation}")

    # 将模型保存成普通的pytorch模型
    torch.save(model_engine.module.state_dict(), 'model.pt')

    model.load_state_dict(torch.load('model.pt'))
    translation = translate(model, source_sentence)
    print(f"翻译: {translation}")


