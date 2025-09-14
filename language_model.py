"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import torch
import torch.nn as nn
import numpy as np


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout, op=''):
        super(WordEmbedding, self).__init__()
        self.op = op

        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
        if 'c' in op:
            self.emb_ = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        # 从np_file加载初始的embedding权重，确认权重的形状
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        # 将初始权重复制到embedding层
        self.emb.weight.data[:self.ntoken] = weight_init

        # 如果提供了TF-IDF和对应的权重
        if tfidf is not None:
            if 0 < tfidf_weights.size:
                # 连接TF-IDF权重与初始权重
                weight_init = torch.cat([weight_init, torch.from_numpy(tfidf_weights)], 0)
            # 使用TF-IDF矩阵乘法调整权重
            weight_init = tfidf.matmul(weight_init) # (N x N') x (N', F)
            if 'c' in self.op:
                #设置权重需要梯度
                self.emb_.weight.requires_grad = True
        if 'c' in self.op:
            #复制调整后的权重到另一个embedding层
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if 'c' in self.op:
            # 如果维度小于 3，则在第二个维度上拼接嵌入
            if len(x.size()) < 3:
                emb = torch.cat((emb, self.emb_(x)), 2)
            else:
                # 如果维度大于等于 3，则在第三个维度上拼接嵌入
                emb = torch.cat((emb, self.emb_(x)), 3)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU if rnn_type == 'GRU' else None

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (weight.new(*hid_shape).zero_(),
                    weight.new(*hid_shape).zero_())
        else:
            return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)
        # self.rnn.flatten_parameters()
        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)
        return output


## 注意：原先此文件包含依赖 lxrt 的 BertEmbedding 实现，但在当前工程中未被使用，且会引入
## `ModuleNotFoundError: No module named 'lxrt'`。为保持最小改动并确保可运行，已移除。