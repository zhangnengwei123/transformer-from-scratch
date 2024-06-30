import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def initialize_weight(x):
    r"""
    初始化权重是确保模型训练效果和效率的重要步骤。
    通过选择适当的初始化方法，可以避免对称性问题、控制梯度的大小，并加速模型的收敛。
    使用PyTorch等框架的初始化函数，可以方便地为模型参数设置合适的初始值，提高训练效果。
    :param x:
    :return:
    """
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


def create_pad_mask(t, pad):
    r"""
    在Encoder中使用Mask, 是为了将Encoder_inputs中没有内容而打上PAD的部分进行Mask, 方便矩阵运算.
    在Decoder中使用Mask, 可能是在Decoder的自注意力对Decoder_inputs的PAD进行Mask,
    也有可能是对Encoder-outputs的PAD进行Mask.

    :param t: [batch_size, seq_len]
    :param pad:
    :return: [batch_size, 1, seq_len]
    """
    mask = t.data.eq(pad).unsqueeze(1)
    return mask


def create_target_self_mask(target):
    r"""
    生成上三角的矩阵

    在Decoder中使用Mask, 在Decoder的自注意力对Decoder_inputs进行Mask,
    :param target:
    :return:[batch, target_len, target_len]
    """
    target_len = target.shape[1]
    ones = torch.ones(target_len, target_len, dtype=torch.uint8)
    self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)
    return self_mask


class MyTransformer(nn.Module):
    r""" A Transformer Model
    The architecture is based on the paper "Attention Is All You Need"
    """

    def __init__(self,
                 i_vocabulary_size,
                 t_vocabulary_size,
                 src_pad_idx,
                 tgt_pad_idx,
                 max_length: int = 12,
                 d_model: int = 512,
                 num_heads=8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 ) -> None:
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(device)

        self.embedding_scale = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.positional_embedding = PositionalEncoding(d_model, max_len=max_length, device=device)

        self.i_vocabulary_embedding = nn.Embedding(i_vocabulary_size, d_model, device=device)
        self.t_vocabulary_embedding = nn.Embedding(t_vocabulary_size, d_model, device=device)

        self.encoder = TransformerEncoder(
            d_model=d_model, dim_feedforward=dim_feedforward,
            dropout=dropout, num_encoder_layers=num_encoder_layers, num_heads=num_heads)

        self.decoder = TransformerDecoder(
            d_model=d_model, dim_feedforward=dim_feedforward,
            dropout=dropout, num_decoder_layers=num_decoder_layers, num_heads=num_heads)

        self.linear = nn.Linear(d_model, t_vocabulary_size, device=device)

    def forward(self, encoder_inputs, decoder_inputs):
        # ------------------------------encode-------------------------------
        # 生成掩码矩阵
        # 假设空白内容占位符的pad_index = 0
        encoder_mask = create_pad_mask(encoder_inputs, self.src_pad_idx)

        # 1 获取输入embedding
        # [batch_size, seq_len] ---> [batch_size, seq_len, d_model]
        encoder_inputs_embedding = self.i_vocabulary_embedding(encoder_inputs)

        # 先处理一次空白字符
        encoder_inputs_embedding.masked_fill_(encoder_mask.squeeze(1).unsqueeze(-1), 0)

        # 对输入的embedding进行缩放（scaling）
        # 对Transformer输入的embedding进行缩放是为了确保数值稳定性，与位置编码匹配，并缓解梯度消失或爆炸问题。
        # 这一步对于Transformer模型的有效训练和性能提升非常重要。
        # 在实现时，通常会将embedding乘以其维度的平方根，以实现上述目标。
        # 2 缩放
        encoder_inputs_embedding *= self.embedding_scale
        # 3 加入positional embedding
        encoder_inputs_positional_embedding = self.positional_embedding(encoder_inputs)
        # 4 +
        encoder_x = encoder_inputs_embedding + encoder_inputs_positional_embedding
        # 5
        # [batch_size, seq_len, d_model]
        encoder_x = self.dropout(encoder_x)

        # ------------------------------decode-------------------------------
        # 4 编码器启动
        encoder_outputs = self.encoder(encoder_x, encoder_mask)

        # 生成掩码矩阵
        decoder_mask = create_pad_mask(decoder_inputs, self.tgt_pad_idx)
        decoder_self_mask = create_target_self_mask(decoder_inputs).to(torch.bool).to(device=self.device)

        encoder_decoder_mask = encoder_mask

        # 1 获取输入embedding
        # [batch_size, seq_len] ---> [batch_size, seq_len, d_model]
        decoder_inputs_embedding = self.i_vocabulary_embedding(decoder_inputs)
        decoder_inputs_embedding.masked_fill_(decoder_mask.squeeze(1).unsqueeze(-1), 0)

        # 2 Shifted right
        decoder_inputs_embedding = decoder_inputs_embedding[:, :-1]
        print(f"decoder_inputs_embedding:{decoder_inputs_embedding.shape}")
        decoder_inputs_embedding = F.pad(decoder_inputs_embedding, (0, 0, 1, 0))

        print(f"decoder_inputs_embedding:{decoder_inputs_embedding.shape}")
        # 3 缩放
        decoder_inputs_embedding *= self.embedding_scale
        # 4 加入positional embedding
        decoder_inputs_positional_embedding = self.positional_embedding(decoder_inputs)
        # 5 +
        decoder_x = decoder_inputs_embedding + decoder_inputs_positional_embedding
        # 6
        decoder_x = self.dropout(decoder_x)

        # 4 解码器启动启动
        decoder_outputs = self.decoder(decoder_x, encoder_outputs,
                                       decoder_self_mask, encoder_decoder_mask)

        # ------------------------
        # 为什么没有论文中最终的softmax层？
        # 在Transformer模型中，解码器的最终输出通常不直接应用Softmax，
        # 这是因为将Softmax计算与损失计算合并可以提高计算效率、增强灵活性，并解决数值稳定性问题。
        # 因此，在训练过程中通常使用CrossEntropyLoss，该损失函数内部已经集成了Softmax计算。
        # 通过这种设计，模型可以更高效地进行训练，并适应不同的应用场景。
        output = self.linear(decoder_outputs)

        return output


class TransformerEncoder(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 dim_feedforward: int = 2048,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_encoder_layers: int = 6
                 ) -> None:
        super().__init__()
        encoder_layers = [TransformerEncoderLayer(d_model, dim_feedforward, dropout, num_heads)
                          for _ in range(num_encoder_layers)]
        self.layers = nn.ModuleList(encoder_layers)

    def forward(self, encoder_inputs, mask):
        encoder_output = encoder_inputs
        for encoder_layer in self.layers:
            encoder_output = encoder_layer(encoder_output, mask)
        return encoder_output


class TransformerDecoder(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_decoder_layers: int = 6,
                 num_heads: int = 8
                 ):
        super().__init__()

        decoder_layers = [TransformerDecoderLayer(d_model, dim_feedforward, dropout, num_heads)
                          for _ in range(num_decoder_layers)]
        self.layers = nn.ModuleList(decoder_layers)

    def forward(self, decoder_input, encoder_output, decoder_self_mask, encoder_decoder_mask):
        decoder_output = decoder_input
        for encoder_layer in self.layers:
            decoder_output = encoder_layer(decoder_input, encoder_output, decoder_self_mask, encoder_decoder_mask)
        return decoder_output


class TransformerEncoderLayer(nn.Module):
    r"""
    TransformerEncoderLayer is made up of self-attention and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".

    """

    def __init__(self,
                 d_model: int = 512,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_heads: int = 8
                 ):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.self_attention_norm = nn.LayerNorm(d_model, eps=1e-6, device=device)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.ffn = FeedForwardNetwork(d_model, dim_feedforward, dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6, device=device)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor,
                src_mask: Tensor
                ) -> Tensor:
        # 按照论文 采用后归一化操作
        # 1.compute self attention
        _x = src
        x = self.self_attention(src, src, src, src_mask)
        # 2.add and norm
        x = self.self_attention_dropout(x)
        x = self.self_attention_norm(x + _x)

        # 3.feedforward
        _x = x
        x = self.ffn(x)
        # 4.add and norm
        x = self.ffn_dropout(x)
        x = self.ffn_norm(x + _x)

        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_heads: int = 8
                 ):
        super().__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.self_attention_norm = nn.LayerNorm(d_model, eps=1e-6, device=device)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.encoder_decoder_self_attention = MultiHeadAttention(d_model=d_model)
        self.encoder_decoder_self_attention_norm = nn.LayerNorm(d_model, eps=1e-6, device=device)
        self.encoder_decoder_self_attention_dropout = nn.Dropout(dropout)

        self.ffn = FeedForwardNetwork(d_model, dim_feedforward, dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6, device=device)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self,
                decoder_input: Tensor,
                encoder_output: Tensor,
                decoder_self_mask,
                decoder_encoder_mask
                ) -> Tensor:
        # 按照论文 采用后归一化操作
        # 1. compute self attention
        _x = decoder_input
        x = self.self_attention(decoder_input, decoder_input, decoder_input, decoder_self_mask)
        # 2. add and norm
        x = self.self_attention_dropout(x)
        x = self.self_attention_norm(x + _x)

        # 3. compute encoder - decoder attention
        _x = x
        x = self.encoder_decoder_self_attention(x, encoder_output, encoder_output, decoder_encoder_mask)
        # 4. add and norm
        x = self.encoder_decoder_self_attention_dropout(x)
        x = self.encoder_decoder_self_attention_norm(x + _x)

        # 5. feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.ffn_dropout(x)
        x = self.ffn_norm(x + _x)

        return x


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int = 512,  # 特征数量
                 num_heads: int = 8,  # 8头
                 dropout: float = 0.1,  # 丢弃
                 ):
        super().__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_heads = num_heads

        self.dim_each_head = d_model // num_heads
        self.d_q = self.d_k = self.d_v = self.dim_each_head

        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False, device=device).cuda()
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False, device=device).cuda()
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False, device=device).cuda()
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False, device=device).cuda()

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, attention_mask):
        r"""
        Attention(Q,K,V) = softmax((Q @ K^T)/sqrt(d_k)) @ V
        :param q:
        :param k:
        :param v:
        :param attention_mask:
        :return:
        """
        # q (batch_size, num_heads, seq_len, d_q)
        # k (batch_size, num_heads, seq_len, d_k)
        # k.transpose(-1, -2) (batch_size, num_heads, seq_len, d_k)  ---> (batch_size, num_heads, d_k, seq_len)
        # torch.matmul(q, k.transpose(-1, -2)) 矩阵相乘 ---> (batch_size, num_heads, seq_len, seq_len)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)  # (batch_size, num_heads, seq_len, seq_len)
        scores.masked_fill_(attention_mask, -1e9)

        attention = nn.Softmax(dim=-1)(scores)  # (batch_size, num_heads, seq_len, seq_len)
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, d_v)
        #  ---> [batch_size, num_heads, seq_len, d_v]
        probabilities = torch.matmul(attention, v)  # [batch_size, num_heads, seq_len, d_v]
        return probabilities, attention

    def forward(self, q, k, v, mask):
        r"""
        To make sure multiple head attention can be used both in encoder and decoder,
        we use Q, K, V respectively.
        input_Q: [batch, len_q, d_model]
        input_K: [batch, len_k, d_model]
        input_V: [batch, len_v, d_model]


        :param q:
        :param k:
        :param v:
        :param mask:
        :return:
        """
        # d_k d_q必然相等 不然无法相乘
        d_q = self.dim_each_head
        d_k = self.dim_each_head
        d_v = self.dim_each_head

        batch_size = q.shape[0]
        seq_len = q.shape[1]
        # 第一步经过线性层 (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        # 第二步经过view (batch_size, seq_len, d_model) --- >  (batch_size, seq_len, num_heads, d_k)
        # 第三步经过转置transpose (batch_size, seq_len, num_heads, d_q) --> (batch_size, num_heads, seq_len, d_q)
        query = self.w_q(q).view(batch_size, seq_len, self.num_heads, d_q).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.num_heads, d_v).transpose(1, 2)

        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # Calculate attention
        x, attention = self.scaled_dot_product_attention(query, key, value, mask)

        # Combine all the heads together
        # (batch_size, num_heads, seq_len, d_k) --> (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2).contiguous()
        # (batch_size, seq_len, num_heads, d_k) --> (batch_size, seq_len, d_model)
        x = x.view(x.shape[0], -1, self.num_heads * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        x = self.w_o(x)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layer1 = nn.Linear(d_model, dim_feedforward, device=device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(dim_feedforward, d_model, device=device)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
