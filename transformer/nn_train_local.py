import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nn_transformer import MyTransformer

# 准备数据
# 先用这个数据作为训练数据，模型训练完成之后，
# 向模型输入这个source_sentence,模型能输出target_sentence 那就很满足。
# 比如输入 你真是个傻逼 可以得到 你真是个小可爱

source_sentences = ["你真是个傻逼", "你咋这么能呢", "我将带头冲锋", "在东南亚打自由搏击", "铃儿响叮当"]
target_sentences = ["你真是个小可爱", "你就是个菜鸡", "我先撤你们上", "这货真牛逼", "今天真开心"]

# 填充空白
sequence_empty_pad = "[PAD]"
# 标记每个输入的开头
sequence_start_pad = "[CLS]"
# 标记每个输入的结束，用于分割语句
sequence_end_pad = "[SEP]"

sequence_max_length = 12

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


# 打印结果
print("字符字典:")
print(char_to_index)

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
                      )

batch_size = 3


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


x, y = get_tokens_batch()


def embedding(tokens_batch):
    return torch.tensor([convert_tokens_to_ids(tokens) for tokens in tokens_batch])


x_embedding = embedding(x)
y_embedding = embedding(y)

logits = model(x_embedding, y_embedding)
# [3,12,50]
print(logits.shape)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)  # 忽略填充值0
# 计算损失
# loss = criterion(outputs, targets) 输入参数的维度是[11,vocabulary_size] [11]
# loss = criterion(logits, target_ids_shifted)

#
decoder_inputs_embedding = y_embedding[:, 1:]
print(f"decoder_inputs_embedding:{decoder_inputs_embedding.shape}")
print(f"decoder_inputs_embedding:{decoder_inputs_embedding}")
decoder_inputs_embedding = F.pad(decoder_inputs_embedding, (0, 1), value=pad_index)
print(f"decoder_inputs_embedding:{decoder_inputs_embedding.shape}")
print(f"decoder_inputs_embedding:{decoder_inputs_embedding}")

decoder_inputs_embedding = decoder_inputs_embedding.contiguous().view(-1)
logits = logits.contiguous().view(-1, vocabulary_size)

print(decoder_inputs_embedding.shape)
print(logits.shape)

# 计算损失
loss = criterion(logits, decoder_inputs_embedding)
print(loss)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 循环训练
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    x, y = get_tokens_batch()
    x_embedding = embedding(x)
    y_embedding = embedding(y)
    logits = model(x_embedding, y_embedding)

    #
    decoder_inputs_embedding = y_embedding[:, 1:]
    print(f"decoder_inputs_embedding:{decoder_inputs_embedding.shape}")
    print(f"decoder_inputs_embedding:{decoder_inputs_embedding}")
    decoder_inputs_embedding = F.pad(decoder_inputs_embedding, (0, 1), value=pad_index)
    print(f"decoder_inputs_embedding:{decoder_inputs_embedding.shape}")
    print(f"decoder_inputs_embedding:{decoder_inputs_embedding}")

    decoder_inputs_embedding = decoder_inputs_embedding.contiguous().view(-1)
    logits = logits.contiguous().view(-1, vocabulary_size)

    print(decoder_inputs_embedding.shape)
    print(logits.shape)

    # 计算损失
    loss = criterion(logits, decoder_inputs_embedding)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 保存模型和优化器状态字典
save_path = "my_first_transformer_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)

print(f"模型已保存到 {save_path}")

# 加载模型
checkpoint = torch.load(save_path)

# 重新初始化模型和优化器
loaded_model = model
loaded_model.load_state_dict(checkpoint['model_state_dict'])

loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.0001)
loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("模型和优化器已加载")

# 使用模型推理

# 设置模型为评估模式
loaded_model.eval()


def translate(model, source_sentence, max_target_length=12):
    # 将源句子标记化并转换为ID
    source_tokens = tokenize(source_sentence)

    source_tokens = ['[CLS]'] + source_tokens + ['[SEP]']
    padding_length_source = max_target_length - len(source_tokens)
    source_tokens = source_tokens + [sequence_empty_pad] * padding_length_source
    print(f"source_tokens:{source_tokens}")

    source_ids = torch.tensor([convert_tokens_to_ids(source_tokens)])
    print(f"source_ids:{source_ids.shape}")

    # attention_mask_source = create_pad_mask(source_ids, pad_index)

    # 初始化解码器输入
    decoder_input_ids = torch.tensor([[cls_token_id]])
    print(decoder_input_ids.shape)

    # 用于存储生成的目标序列
    generated_ids = []

    for _ in range(max_target_length):
        # 前向传播
        with torch.no_grad():
            logits = loaded_model(source_ids, decoder_input_ids)

        # 获取当前时间步的预测结果
        # [1, 21128]
        next_token_logits = logits[:, -1, :]
        # 找出概率最大的哪一个
        next_token_id = next_token_logits.argmax(dim=-1).item()
        print(f"next_token_id : {next_token_id}")

        # 添加到生成的序列中
        generated_ids.append(next_token_id)

        # 更新解码器输入
        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[next_token_id]])], dim=-1)
        print(decoder_input_ids)
        # 如果预测到了[SEP]标记，则停止生成
        if next_token_id == seq_token_id:
            break

    # 转换生成的ID为标记
    generated_tokens = convert_ids_to_tokens(generated_ids)
    return generated_tokens


# 测试
source_sentence = "你真是个傻逼"
translation = translate(loaded_model, source_sentence)
print(f"翻译: {translation}")
