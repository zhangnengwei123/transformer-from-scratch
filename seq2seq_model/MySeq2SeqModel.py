from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim

bert_model_id = "/Users/zhangnengwei/nnnewworkspace/bert-base-chinese"
# 使用BERT的预训练中文词表
tokenizer_cn = BertTokenizer.from_pretrained(bert_model_id)

# 准备数据
# 先用这个数据作为训练数据，模型训练完成之后，
# 向模型输入这个source_sentence,模型能输出target_sentence 那就很满足。
source_sentence = "你真是个傻逼"
target_sentence = "你真是个小可爱"

# 将句子分成一个一个字符
# 标记化并添加特殊标记
# 为啥要添加特殊标记呢？
# 是为了模型可以识别需要的数据。后续还需要对数据进行补全，使得数据规整，模型得以训练。
source_tokens = tokenizer_cn.tokenize(source_sentence)
source_tokens = ['[CLS]'] + source_tokens + ['[SEP]']
# source_tokens ['[CLS]', '你', '真', '是', '个', '傻', '逼', '[SEP]']

target_tokens = tokenizer_cn.tokenize(target_sentence)
target_tokens = ['[CLS]'] + target_tokens + ['[SEP]']
# target_tokens ['[CLS]', '你', '真', '是', '个', '小', '可', '爱', '[SEP]']

# 进行embedding 转换成向量表示
# 为啥要转换成数字呢？
# 为了模型认识。
# 相当于一个密码本，模型能够通过这个密码本进行编码解码。
source_ids = tokenizer_cn.convert_tokens_to_ids(source_tokens)
target_ids = tokenizer_cn.convert_tokens_to_ids(target_tokens)
# source_ids [101, 872, 4696, 3221, 702, 1004, 6873, 102]
# target_ids [101, 872, 4696, 3221, 702, 2207, 1377, 4263, 102]

# 确定最大长度（例如设置为12）
# 为啥设置这个最大长度？
# 为了整齐划一，模型可以训练.
max_source_length = 12
max_target_length = 12

# 填充（PAD）并创建注意力掩码
padding_length_source = max_source_length - len(source_ids)
source_ids = source_ids + [0] * padding_length_source
attention_mask_source = [1] * len(source_tokens) + [0] * padding_length_source
# [101, 872, 4696, 3221, 702, 1004, 6873, 102, 0, 0, 0, 0]
# [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

padding_length_target = max_target_length - len(target_ids)
target_ids = target_ids + [0] * padding_length_target
attention_mask_target = [1] * len(target_tokens) + [0] * padding_length_target
# [101, 872, 4696, 3221, 702, 2207, 1377, 4263, 102, 0, 0, 0]
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

# 转换为PyTorch张量
source_ids_tensor = torch.tensor([source_ids])  # [1, 12]
attention_mask_source_tensor = torch.tensor([attention_mask_source])  # [1, 12]

target_ids_tensor = torch.tensor([target_ids])  # [1, 12]
attention_mask_target_tensor = torch.tensor([attention_mask_target])  # [1, 12]


# 定义模型
class Seq2SeqModel(nn.Module):
    def __init__(self, decoder_vocab_size):
        super().__init__()
        # 现在只看数据流的情况下，直接使用bert模型的编码器和解码器
        encoder = BertModel.from_pretrained(bert_model_id)
        decoder = BertModel.from_pretrained(bert_model_id)
        d_model = decoder.config.hidden_size  # 隐藏层的维度，也就是模型中间分析的特征数量

        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(d_model, decoder_vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 编码器的输出
        encoder_hidden_states = encoder_outputs.last_hidden_state
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
                                       encoder_hidden_states=encoder_hidden_states)
        # 解码器的输出
        return self.linear(decoder_outputs.last_hidden_state)


# 初始化模型
model = Seq2SeqModel(tokenizer_cn.vocab_size)

# 先跑一次看一下中间过程
# 前向传播
logits = model(input_ids=source_ids_tensor,
               attention_mask=attention_mask_source_tensor,
               # 将最后一位干掉，让模型去训练，生成这一位。
               decoder_input_ids=target_ids_tensor[:, :-1],
               decoder_attention_mask=attention_mask_target_tensor[:, :-1])

# [1, 11, 21128]
# 模型输出在字典上的每个字符都有一个概率

# 调整目标张量形状以适应损失函数
# contiguous() 方法在 PyTorch 中的作用是确保张量的内存布局是连续的，
# 从而使得一些需要连续内存布局的操作（如 view）能够顺利进行。
# 了解和正确使用 contiguous() 是处理复杂张量操作时的一项重要技能。
# .view(-1)是一种常用的方法，用于将张量展平（flatten）成一个一维张量。
target_ids_shifted = target_ids_tensor[:, 1:].contiguous().view(-1)
logits = logits.contiguous().view(-1, tokenizer_cn.vocab_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充值0
# 计算损失
# loss = criterion(outputs, targets) 输入参数的维度是[11,vocabulary_size] [11]
loss = criterion(logits, target_ids_shifted)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 循环训练
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    logits = model(input_ids=source_ids_tensor,
                   attention_mask=attention_mask_source_tensor,
                   decoder_input_ids=target_ids_tensor[:, :-1],
                   decoder_attention_mask=attention_mask_target_tensor[:, :-1])

    # 调整目标张量形状以适应损失函数
    # contiguous() 方法在 PyTorch 中的作用是确保张量的内存布局是连续的，
    # 从而使得一些需要连续内存布局的操作（如 view）能够顺利进行。
    # 了解和正确使用 contiguous() 是处理复杂张量操作时的一项重要技能。
    # .view(-1)是一种常用的方法，用于将张量展平（flatten）成一个一维张量。
    target_ids_shifted = target_ids_tensor[:, 1:].contiguous().view(-1)
    logits = logits.contiguous().view(-1, tokenizer_cn.vocab_size)

    # 计算损失
    loss = criterion(logits, target_ids_shifted)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 保存模型和优化器状态字典
save_path = "seq2seq_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)

print(f"模型已保存到 {save_path}")

# 加载模型
checkpoint = torch.load(save_path)

# 重新初始化模型和优化器
loaded_model = Seq2SeqModel(tokenizer_cn.vocab_size)
loaded_model.load_state_dict(checkpoint['model_state_dict'])

loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.0001)
loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("模型和优化器已加载")

# 使用模型推理

# 设置模型为评估模式
loaded_model.eval()

def translate(model, source_sentence, max_target_length=12):
    # 将源句子标记化并转换为ID
    source_tokens = tokenizer_cn.tokenize(source_sentence)
    source_tokens = ['[CLS]'] + source_tokens + ['[SEP]']
    source_ids = tokenizer_cn.convert_tokens_to_ids(source_tokens)

    # 填充源输入
    padding_length_source = max_source_length - len(source_ids)
    source_ids = source_ids + [0] * padding_length_source
    attention_mask_source = [1] * len(source_tokens) + [0] * padding_length_source

    # 转换为PyTorch张量
    source_ids_tensor = torch.tensor([source_ids])
    attention_mask_source_tensor = torch.tensor([attention_mask_source])

    # 初始化解码器输入
    decoder_input_ids = torch.tensor([[tokenizer_cn.cls_token_id]])

    # 用于存储生成的目标序列
    generated_ids = []

    for _ in range(max_target_length):
        # 前向传播
        with torch.no_grad():
            logits = model(input_ids=source_ids_tensor,
                           attention_mask=attention_mask_source_tensor,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=torch.ones_like(decoder_input_ids))

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
        if next_token_id == tokenizer_cn.sep_token_id:
            break

    # 转换生成的ID为标记
    generated_tokens = tokenizer_cn.convert_ids_to_tokens(generated_ids)
    return tokenizer_cn.convert_tokens_to_string(generated_tokens)


# 测试
source_sentence = "她真是个傻逼"
translation = translate(loaded_model, source_sentence)
print(f"翻译: {translation}")
