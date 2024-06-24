import torch
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# # CPU表现更好，因为维度简单
# torch_rand1 = torch.rand(10000,10000).to(device)
# torch_rand2 = torch.rand(10000,10000).to(device)
# np_rand1 = torch.rand(10000,10000)
# np_rand2 = torch.rand(10000,10000)
#
# GPU表现更好
torch_rand1 = torch.rand(100, 100, 100, 100).to(device)
torch_rand2 = torch.rand(100, 100, 100, 100).to(device)
np_rand1 = torch.rand(100, 100, 100, 100)
np_rand2 = torch.rand(100, 100, 100, 100)

start_time = time.time()
rand = (torch_rand1 @ torch_rand2)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"{elapsed_time:.8f}")

start_time = time.time()
rand1 = np.multiply(np_rand1, np_rand2)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"{elapsed_time:.8f}")

# define a probability tensor
probabilities = torch.tensor([0.1, 0.9])

samples = torch.multinomial(probabilities, num_samples=10, replacement=True)
print(samples)

tensor = torch.tensor([1, 2, 3, 4])
out = torch.cat((tensor, torch.tensor([5])), dim=0)
print(out)

out = torch.tril(torch.ones(5, 5))
print(out)

out = torch.triu(torch.ones(5, 5))
print(out)

# 遮盖
out = torch.zeros(5, 5).masked_fill(torch.tril(torch.ones(5, 5)) == 0, float('-inf'))
print(out)

torch.exp(out)

# 转置
input = torch.zeros(2, 3, 4)

B = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
tensor3 = torch.tensor([7, 8, 9])

stacked_tensor = torch.stack([tensor1, tensor2, tensor3])

import torch.nn as nn

sample = torch.tensor([10., 10., 10.])
linear = nn.Linear(3, 3, bias=False)
print(linear(sample))

import torch.nn.functional as F

tensor1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

softmax_output = F.softmax(tensor, dim=0)

print(softmax_output)

import torch
import torch.nn as nn

# 定义嵌入层
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)

# 输入是一个索引序列
input_indices = torch.tensor([1, 2, 3, 4])

# 获取嵌入向量
embedded_vectors = embedding(input_indices)

print("输入索引:", input_indices)
print("嵌入向量:\n", embedded_vectors)

a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(torch.matmul(a, b))
