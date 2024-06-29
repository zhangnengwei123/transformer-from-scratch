import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# Block Size（块大小）是指在处理文本时，模型一次读取或生成的最大文本长度。
block_size = 3
# Batch Size（批大小）是指在一次训练迭代中，模型同时处理的样本数。
batch_size = 4

learning_rate = 1e-3
max_iters = 100
eval_iters = 10
n_embd = 24

n_layer = 4
n_head = 3

dropout = 0.2


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape(-1) ** -0.5
        wei = wei.masked_fill(self.tril[:T:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        # GPU并行
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        # 每个头需要获取的特征数量
        head_size = n_embd // n_head
        # 多头注意力机制
        self.sa = MultiHeadAttention(n_head, head_size)
        # 前馈
        self.ffwd = FeedForward(n_embd)
        # 归一化
        self.ln1 = nn.LayerNorm(n_embd)
        # 归一化
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class MyGptLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 前向传播
    def forward(self, input_index, targets):
        B, T = input_index.shape

        # 获取输入embedding position_embedding
        tok_emb = self.token_embedding_table(input_index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        N1, N2, C = logits.shape
        N = N1 * N2
        logits = logits.view(N, C)
        targets = targets.view(N)
        # 交叉熵函数 input(N,C) targets(N)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

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
