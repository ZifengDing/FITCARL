import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable
from collections import OrderedDict


class AttentionSelectContext(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super(AttentionSelectContext, self).__init__()
        self.Bilinear = nn.Bilinear(dim, dim, 1, bias=False)
        self.Linear_tail = nn.Linear(dim, dim, bias=False)
        self.Linear_head = nn.Linear(dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def intra_attention(self, head, rel, tail, mask):
        """

        :param head: [b, dim]
        :param rel: [b, max, dim]
        :param tail:
        :param mask:
        :return:
        """
        head = head.unsqueeze(1).repeat(1, rel.size(1), 1)
        score = self.Bilinear(head, rel).squeeze(2)

        score = score.masked_fill_(mask, -np.inf)
        att = torch.softmax(score, dim=1).unsqueeze(dim=1)  # [b, 1, max]

        head = torch.bmm(att, tail).squeeze(1)
        return head

    def forward(self, left, right, mask_left=None, mask_right=None):
        """
        :param left: (head, rel, tail)
        :param right:
        :param mask_right:
        :param mask_left:
        :return:
        """
        head_left, rel_left, tail_left = left
        head_right, rel_right, tail_right = right
        weak_rel = head_right - head_left

        left = self.intra_attention(weak_rel, rel_left, tail_left, mask_left)
        right = self.intra_attention(weak_rel, rel_right, tail_right, mask_right)

        left = torch.relu(self.Linear_tail(left) + self.Linear_head(head_left))
        right = torch.relu(self.Linear_tail(right) + self.Linear_head(head_right))

        left = self.dropout(left)
        right = self.dropout(right)

        left = self.layer_norm(left + head_left)
        right = self.layer_norm(right + head_right)
        return left, right


class TimeEncode(nn.Module):
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.time_dim = dim
        self.basis_freq = torch.nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts):
        '''
        ts: [B, L]
        '''
        B = ts.size(0)
        L = ts.size(1)
        map_ts = ts.unsqueeze(2) * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention
    """

    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None, add_attn=None):
        """
        :param attn_mask: [batch, time]
        :param scale:
        :param q: [batch, time, dim]
        :param k: [batch, time, dim]
        :param v: [batch, time, dim]
        :return:
        """
        attn = torch.bmm(q, k.transpose(1, 2))  # [B, seq_len, seq_len]

        if scale:
            attn = attn * scale
        if add_attn is not None:
            attn += add_attn
        if attn_mask:
            attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # [B, seq_len, D]
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Implement without batch dim"""

    def __init__(self, time_encoder, time_dim, model_dim, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

        # self.time_encoder = TimeEncode(model_dim)
        self.time_encoder = time_encoder

        self.time_weight = nn.Parameter(torch.ones(time_dim) * 0.001)
        # nn.init.xavier_uniform_(self.time_weight.data)  # xavier_uniform_ not for 1-dim vector

    def forward(self, query, key, value, attn_mask=None, t_seq=None):
        """
        To be efficient, multi- attention is cal-ed in a matrix totally
        :param attn_mask:
        :param query: [batch, time, per_dim * num_heads]
        :param key:
        :param value:
        :return: [b, t, d*h]
        """
        residual = query
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        scale = (key.size(-1) // self.num_heads) ** -0.5

        t_diff_attn = self.time_aware_att(t_seq).repeat(self.num_heads, 1, 1)
        context, attn = self.dot_product_attention(query, key, value, scale,
                                                   attn_mask, add_attn=t_diff_attn)  # [B * H, S, d], d for dim_per_head
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output, attn

    def time_aware_att(self, t_seq):
        '''
        t_seq: [B, S]
        '''
        B = t_seq.shape[0]
        S = t_seq.shape[1]
        t_diff = t_seq.unsqueeze(-1) - t_seq.unsqueeze(1)  # i - j, [B, S, S],  [B, S, 1] - [B, 1, S]
        t_diff_emb = self.time_encoder.forward_transformer(t_diff.reshape(B, -1)).reshape(B, S, S, -1)  # [B, S * S, D] -> [B, S, S, D]
        t_diff_mat = torch.matmul(t_diff_emb, self.time_weight)  # [B, S, S]
        return t_diff_mat


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, batch_len, seq_len):
        """
        :param batch_len: scalar
        :param seq_len: scalar
        :return: [batch, time, dim]
        """
        input_pos = torch.tensor([list(range(1, seq_len + 1)) for _ in range(batch_len)]).to(
            self.position_encoding.weight.device)
        return self.position_encoding(input_pos)


class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.gelu = GELU()

    def forward(self, x):
        """

        :param x: [b, t, d*h]
        :return:
        """
        output = x.transpose(1, 2)  # [b, d*h, t]
        output = self.w2(self.gelu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, time_encoder, time_dim, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        # ffn_dim
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(time_encoder, time_dim, model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None, t_seq=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask, t_seq=t_seq)
        output = self.feed_forward(context)
        return output, attention


class TransformerEncoder(nn.Module):
    def __init__(self, time_encoder, time_dim, model_dim=100, ffn_dim=800, num_heads=4, dropout=0.1, num_layers=6, max_seq_len=3,
                 with_pos=True):
        super(TransformerEncoder, self).__init__()
        self.with_pos = with_pos
        self.num_heads = num_heads

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(time_encoder, time_dim, model_dim * num_heads, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.cls_embed = nn.Parameter(torch.rand(1, model_dim), requires_grad=True)
        self.compress = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(model_dim * num_heads, model_dim)),
            ('bn', nn.BatchNorm1d(max_seq_len)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=0.5)),
        ]))

        self.model_dim = model_dim

    def repeat_dim(self, emb):
        """
        :param emb: [batch, t, dim]
        :return:
        """
        return emb.repeat(1, 1, self.num_heads)

    def forward(self, support_emb_mat, query_t, support_t):
        """
        :param left: [batch, dim]
        :param right: [batch, dim]
        :param support_emb_l: [B, F, D]
        :return:
        """
        B, F, D = support_emb_mat.shape
        cls_emb = self.cls_embed.expand_as(support_emb_mat[:, 0, :]).unsqueeze(1)  # [B, 1, D]
        seq = torch.cat([cls_emb, support_emb_mat], dim=1)
        # pos = self.pos_embedding(batch_len=B, seq_len=4)
        if self.with_pos:
            output = seq
            # output = seq + pos
        else:
            output = seq
        output = self.repeat_dim(output)  # [B, F+1, D * H]
        output = output.unsqueeze(1).repeat(1, query_t.shape[1], 1, 1).reshape(B * query_t.shape[1], F + 1, -1)
        attentions = []
        t_seq = self.build_time_sequence(query_t, support_t).to(self.cls_embed.device)

        for encoder in self.encoder_layers:
            output, attention = encoder(output, t_seq=t_seq)
            attentions.append(attention)
        output = self.compress(output)
        return output[:, 0, :].reshape(B, -1, self.model_dim)

    @staticmethod
    def build_time_sequence(query_t, support_t):
        '''
        quert_t: [B, Q]
        support_t: [B, F]
        '''
        B, Q = query_t.shape
        _, F = support_t.shape
        query_t = torch.unsqueeze(query_t, -1)  # [B, Q, 1]
        support_t = torch.unsqueeze(support_t, 1).expand(-1, Q, -1)  # [B, 1, F] -> [B, Q, F]
        time_seq = torch.cat([query_t, support_t], dim=-1)  # [B, Q, 1 + F]
        return time_seq.reshape(B * Q, F + 1)


class SoftSelectAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftSelectAttention, self).__init__()

    def forward(self, support, query):
        """
        :param support: [few, dim]
        :param query: [batch, dim]
        :return:
        """
        query_ = query.unsqueeze(1).expand(query.size(0), support.size(0), query.size(1)).contiguous()  # [b, few, dim]
        support_ = support.unsqueeze(0).expand_as(query_).contiguous()  # [b, few, dim]

        scalar = support.size(1) ** -0.5  # dim ** -0.5
        score = torch.sum(query_ * support_, dim=2) * scalar
        att = torch.softmax(score, dim=1)

        center = torch.mm(att, support)
        return center


class SoftSelectPrototype(nn.Module):
    def __init__(self, r_dim):
        super(SoftSelectPrototype, self).__init__()
        self.Attention = SoftSelectAttention(hidden_size=r_dim)

    def forward(self, support, query):
        center = self.Attention(support, query)
        return center
