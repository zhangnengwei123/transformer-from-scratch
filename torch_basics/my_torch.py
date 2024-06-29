import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Block Size（块大小）是指在处理文本时，模型一次读取或生成的最大文本长度。
block_size = 8
# Batch Size（批大小）是指在一次训练迭代中，模型同时处理的样本数。
batch_size = 4

with open('../bigraml_model/ebook_free.text', 'r', encoding='UTF-8') as f:
    text = f.read()
print(len(text))
print(text[:200])  # 打印前200个字符

chars = sorted(set(text))
print(len(chars))
print(chars)
vocabulary_size = len(chars)  # 词汇表大小

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


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    print(ix)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


x, y = get_batch('train')
print('inputs:')
print(x)
print('targets:')
print(y)

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print("when input is", context, "target is", target)
