import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# Block Size（块大小）是指在处理文本时，模型一次读取或生成的最大文本长度。
block_size = 8
# Batch Size（批大小）是指在一次训练迭代中，模型同时处理的样本数。
batch_size = 4

learning_rate = 1e-3
max_iters = 100
eval_iters = 10

with open('ebook_free.text', 'r', encoding='UTF-8') as f:
    text = f.read()
print(len(text))
print(text[:200])  # 打印前200个字符

# 将每个字符找出来，做成词汇表
chars = sorted(set(text))
print(len(chars))
print(chars)
vocabulary_size = len(chars)  # 词汇表大小

# 制作编码器解码器
# for i, ch in enumerate(chars):
#     print(f"{i}:{ch}")

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: "".join([int_to_string[i] for i in l])

print(encode("hello"))
print(decode([63, 60, 67, 67, 70]))

# 使用torch处理数据
data = torch.tensor(encode(text), dtype=torch.long)
print(data[:100])

# 分成训练集 验证集
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

# 通过现在的输入x预测y，看看怎么回事吧
x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print("when input is", context, "target is", target)


# 引入batch size进行并行处理 提升性能
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 生成batch_size个随机数
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


x, y = get_batch('train')
print('inputs:')
print(x)
print('targets:')
print(y)

# 创建embedding表
token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

# 获取input
input_logits = token_embedding_table(x)
N1, N2, C = input_logits.shape
# [4,8,96]
# [batch_size,block_size,vocabulary_size]

# 准备交叉熵函数的输入
input_logits = input_logits.view(N1 * N2, C)
targets = y.view(N1 * N2)

# 计算loss
# 交叉熵函数 input(N,C) targets(N)
loss = F.cross_entropy(input_logits, targets)


# 有了这个loss之后就可以通过优化器进行循环优化 反向传播
# 接下来实现模型


class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    # 前向传播
    def forward(self, input_index, targets):
        input_logits = self.token_embedding_table(input_index)
        N1, N2, C = input_logits.shape
        N = N1 * N2
        input = input_logits.view(N, C)
        targets = targets.view(N)
        # 交叉熵函数 input(N,C) targets(N)
        loss = F.cross_entropy(input, targets)
        return input, loss

    # 获取logits
    def get_logits(self, input_index):
        return self.token_embedding_table(input_index)

    # 生成预测文本
    def generate(self, input_index, max_new_tokens):
        # input_index是当前上下文（B,T）数组的下标
        for _ in range(max_new_tokens):
            # get the predictions
            logits = self.get_logits(input_index)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            index_next = torch.multinomial(probs, num_samples=1)
            input_index = torch.cat((input_index, index_next), dim=1)  # (B,T+1)
        return input_index


model = BigramLanguageModel(vocabulary_size)

model.forward(x, y)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(model.generate(context, max_new_tokens=50)[0].tolist())
print(generated_chars)


@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

max_iters=1000
eval_iters = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step:{iter},loss:{losses}")

    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(model.generate(context, max_new_tokens=50)[0].tolist())
print(generated_chars)
