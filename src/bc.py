"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from src.fc import FCNet


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2,.5], k=1):   # k = 3
        super(BCNet, self).__init__()
        
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1]) # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        
        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim*self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v).transpose(1,2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) # b x h_dim x v x q
            logits = d_.transpose(1,2).transpose(2,3) # b x v x q x h_dim
            return logits

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        # epoch 1, time: 157.84
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat # broadcast, b x h_out x v x h_dim
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2,3)) # b x h_out x v x q
            logits = logits + self.h_bias
            return logits # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        # epoch 1, time: 304.87
        else: 
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        #transpose 函数用于交换张量的两个维度
        #unsqueeze 函数用于在指定位置增加一个维度，这个新维度的大小为1
        v_ = self.v_net(v).transpose(1,2).unsqueeze(2) # b x d x 1 x v
        q_ = self.q_net(q).transpose(1,2).unsqueeze(3) # b x d x q x 1
        # v_.float() 和 q_.float()：将张量 v_ 和 q_ 转换为浮点数类型，以进行矩阵乘法。
        # w.unsqueeze(1)：在 w 张量的第二维增加一个维度，大小为1。这样做是为了使 w 的形状与 v_ 张量的形状兼容，以便进行矩阵乘法。
        # torch.matmul(v_, w.unsqueeze(1))：首先计算v_和w的矩阵乘法。这个操作将v_的每个batch与w进行外积，可能用于计算加权特征。
        # torch.matmul(..., q_)：接着，将上一步的结果与 q_ 进行矩阵乘法。这可以视为一种查询操作，将加权特征与问题特征 q_ 进行交互，以生成最终的输出
        # .type_as(v_)：将结果 logits 的数据类型转换为与 v_ 相同，确保数据类型的一致性。
        logits = torch.matmul(torch.matmul(v_.float(), w.unsqueeze(1).float()), q_.float()).type_as(v_) # b x d x 1 x 1
        # logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)# b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2) # b x d
        if 1 < self.k:
            logits = logits.unsqueeze(1) # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k # sum-pooling
        return logits

    def forward_with_v_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2)  # b x d x v
        q_ = self.q_net(q).transpose(1, 2)  # b x d x q
        logits = torch.cat([torch.matmul(q_, w.transpose(1, 2).float()), v_.float()], 1).type_as(v_)  # b x 2xd x v
        # logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)# b x d x 1 x 1
        logits = logits.transpose(1, 2)
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits

    def forward_with_q_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2)  # b x d x v
        q_ = self.q_net(q).transpose(1, 2)  # b x d x q
        logits = torch.mul(torch.matmul(v_, w.float()), q_.float()).type_as(v_)  # b x 2xd x v
        # logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)# b x d x 1 x 1
        logits = logits.transpose(1, 2)
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits

if __name__=='__main__':
    net = BCNet(1024,1024,1024,1024).cuda()
    x = torch.Tensor(512,36,1024).cuda()
    y = torch.Tensor(512,14,1024).cuda()
    out = net.forward(x,y)
