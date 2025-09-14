"""
Coarse to Fine Adaption Flow
HuyTran
https://arxiv.org/abs/1805.07932

This code is written by Huy Tran.
"""
import logging

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from ISubGVQA.models.build import build_model
from src.attention import BiAttention
from language_model import WordEmbedding, QuestionEmbedding
from src.classifier import SimpleClassifier
from src.fc import FCNet
from src.bc import BCNet
from src.counting import Counter
from src.utils import tfidf_loading
from src.FFOE.dataset import Dictionary, GQAFeatureDataset ,VQAFeatureDataset
from xvlm.model_vqa import XVLM
import numpy as np
from aoa_pytorch import AoA

MAX_VQA_LENGTH = 20

class BanFusion(nn.Module):
    def __init__(self, dataset, b_att, b_net, q_prj, c_prj, counter, gamma, omega):
        super(BanFusion, self).__init__()
        self.dataset = dataset
        self.glimpse = gamma
        self.omega = omega
        self.v_att = b_att
        self.b_net = nn.ModuleList(b_net)   # 将b模态的网络层列表转换为ModuleList，以便PyTorch可以追踪其参数。
        self.q_prj = nn.ModuleList(q_prj)
        if counter is not None:  # if do not use counter
            self.c_prj = nn.ModuleList(c_prj)
        self.counter = counter

    def forward(self, mod1, b, mod2):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        if self.counter is not None:
            boxes = b[:, :, :4].transpose(1, 2)
        b_emb = [0] * self.glimpse  # 初始化一个列表，用于存储每个残差连接的b模态嵌入。
        att, logits = self.v_att.forward_all(mod1, mod2)  # b x g x v x q

        for g in range(self.glimpse):
            # 使用b模态网络层和双注意力模块的权重来获取b模态的嵌入。
            b_emb[g] = self.b_net[g].forward_with_weights(mod1, self.omega * mod2, att[:, g, :, :])
            mod1 = self.omega * self.q_prj[g](b_emb[g].unsqueeze(1)) * mod1 + mod1

            if self.counter is not None:
                atten, _ = logits[:, g, :, :].max(2)
                embed = self.counter(boxes, atten)

            if self.counter is not None:
                mod1 = mod1 + self.c_prj[g](embed).unsqueeze(1)
        return mod1


class CFRF_Model(nn.Module):
    def __init__(self, dataset, args, XVLM_model,ISubGVQA_model, w_emb, q_emb, sw_emb, s_emb, ew_emb, qe_joint, vs_joint, vq_joint,
                 classifier, gamma):
        super(CFRF_Model, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.sw_emb = sw_emb
        self.s_emb = s_emb
        self.ew_emb = ew_emb
        self.qe_joint = qe_joint
        self.vs_joint = vs_joint
        self.vq_joint = vq_joint
        self.gamma = gamma
        # self.lxmert_encoder = lxmert_encoder
        self.XVLM_model = XVLM_model
        self.ISubGVQA_model = ISubGVQA_model
        #nn.Parameter 表示这个张量是模型的一个参数，并且在训练过程中会被优化
        self.adapted_w = nn.Parameter(torch.ones(3, dataset.num_ans_candidates))  # adapted_w 的形状是 (2, num_ans_candidates)
        self.classifier = classifier
        self.attn_AOA_fine = AoA(dim = 1024,heads = 16)
        self.layers = args.AOA_layers
        self.attn_AOA_fine = nn.ModuleList([AoA(dim=1024, heads=16) for _ in range(self.layers)])
        
        # 性能优化相关
        self.args = args
        self.use_memory_efficient_attention = getattr(args, 'memory_efficient_attention', False)
        self.gradient_checkpointing = getattr(args, 'gradient_checkpointing', False)
        
        # 启用梯度检查点以节省内存
        if self.gradient_checkpointing:
            self.enable_gradient_checkpointing()
    
    def enable_gradient_checkpointing(self):
        """启用梯度检查点以节省内存"""
        if hasattr(self.XVLM_model, 'gradient_checkpointing_enable'):
            self.XVLM_model.gradient_checkpointing_enable()
        if hasattr(self.ISubGVQA_model, 'gradient_checkpointing_enable'):
            self.ISubGVQA_model.gradient_checkpointing_enable()
    
    def optimize_memory_usage(self):
        """优化内存使用"""
        # 将不常用的模块移到CPU
        if hasattr(self, 'w_emb') and self.w_emb is not None:
            self.w_emb = self.w_emb.cpu()
        if hasattr(self, 'sw_emb') and self.sw_emb is not None:
            self.sw_emb = self.sw_emb.cpu()
        if hasattr(self, 'ew_emb') and self.ew_emb is not None:
            self.ew_emb = self.ew_emb.cpu()
    
    def forward_with_memory_optimization(self, v, b, w, e, q, image, question_xvlm, answer_input, scene_graphs, question_ISubGVQA):
        """内存优化的前向传播"""
        device = v.device
        
        # 将CPU上的嵌入移到GPU
        if hasattr(self, 'w_emb') and self.w_emb is not None:
            self.w_emb = self.w_emb.to(device)
        if hasattr(self, 'sw_emb') and self.sw_emb is not None:
            self.sw_emb = self.sw_emb.to(device)
        if hasattr(self, 'ew_emb') and self.ew_emb is not None:
            self.ew_emb = self.ew_emb.to(device)
        
        # 执行前向传播
        result = self.forward(v, b, w, e, q, image, question_xvlm, answer_input, scene_graphs, question_ISubGVQA)
        
        # 将嵌入移回CPU以节省GPU内存
        self.optimize_memory_usage()
        
        return result


    #[v, b, w, e, q, ,image, question_xvlm,answer_input]
    # def forward(self, v, b, w, e, q, image, question_xvlm, answer_input,scene_graphs,question_ISubGVQA):
    # def forward(self, image, question_xvlm, answer_input, scene_graphs, question_ISubGVQA):
    def forward(self, v, b, w, e, q, image, question_xvlm, answer_input,scene_graphs,question_ISubGVQA, answers=None):
        # v (batch_size,50,2048)    视觉特征
        # b (batch_size,50,6)       类似于ax+b 中的b
        # q (batch_size,12)         问题特征
        # e (batch_size,7)          实体特征
        # w (batch_size,30)
        # s      ['Who is holding the umbrella?',......., 'Is the rug to the left of a sofa?']

        #scene_graph: DataBatch(x=[142, 4], edge_index=[2, 685], edge_attr=[685], added_sym_edge=[48], x_bbox=[142, 4],batch=[142], ptr=[9])

        # print("^^^^^^^^^^^^^^^^^^^^^begin^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print("^^^^^^^^^^^^^^^^^^^^^end^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # # 抛出异常并传递异常信息
        # try:
        #     x = 1 / 0
        # except ZeroDivisionError as e:
        #     raise RuntimeError("上面是f的内容") from e
        # print("scene_graphs:")
        # print(scene_graphs)

        ISubGVQA_output = self.ISubGVQA_model(
            node_embeddings=scene_graphs.x,
            edge_index=scene_graphs.edge_index,
            edge_embeddings=scene_graphs.edge_attr,
            batch=scene_graphs.batch,
            questions=question_ISubGVQA["input_ids"],
            qsts_att_mask=question_ISubGVQA["attention_mask"],
            return_masks=True,
            scene_graphs=scene_graphs,
        )
        (ISubGVQA_logist,_,_,_,_) = ISubGVQA_output     #ISubGVQA_logist [8,1533]
        ISubGVQA_logist = F.softmax(ISubGVQA_logist, dim=-1)

        '''粗粒度学习'''
        # intra-inter Modality interaction
        # Question_Coares + Visual_Coares
        """
        Visual_Coares: v(batch_size,50,2048)
        Question_Coares: s ['Who is holding the umbrella?',......., 'Is the rug to the left of a sofa?']
        """
        # Lxmert encoder
        # lxmert_logit -->(64,1533)
        # lxmert_qemb --> ([batch, 20, 768])
        # lxmert_v  -->  ([batch, 50, 768])
        # lxmert_qemb, lxmert_v, lxmert_logit = self.lxmert_encoder(v, b, s)  # 粗粒度学习



        # 在训练阶段也直接返回 XVLM 对全答案集的概率，用于融合/蒸馏；默认不对 XVLM 求梯度（with_grad=False）
        xvlm_probs = self.XVLM_model(
            image,
            question_xvlm,
            answer_input,
            train=self.training,
            return_probs=True,
            with_grad=getattr(self.args, 'tune_xvlm', False),
            k=128,
        )

        # XVLM 原生 CE（方案B）：仅在训练且开启 --tune_xvlm 时计算
        xvlm_ce_loss = None
        if self.training and getattr(self.args, 'tune_xvlm', False) and answers is not None:
            try:
                k_int = getattr(self.args, 'topk', 6)
            except Exception:
                k_int = 6
            xvlm_ce_loss = self.XVLM_model(
                image,
                question_xvlm,
                answer_input,
                train=True,
                return_probs=False,
                weights=answers,
                k=k_int,
            )

        # logging.debug(f'---------------------------------------')
        # logging.debug(f'base_model (topk_ids) :{topk_ids}')
        # logging.debug(f'base_model (topk_ids.shape) :{topk_ids.shape}')   #([8, 128])

        # logging.debug(f'base_model (topk_probs) :{topk_probs}')   #[0] [8.5342e-01,1.4657e-01, 3.6297e-06,...,5.0879e-15,3.9270e-15,2.1372e-15]
        # logging.debug(f'base_model (topk_probs) :{topk_probs.shape}')

        '''信息过滤 +  细粒度学习'''

        # Question embedding
        w_emb = self.w_emb(q)  # (batch_size,12,600)
        q_emb = self.q_emb.forward_all(w_emb)  # (batch_size,12,1024)

        # Semantic embedding
        sw_emb = self.sw_emb(w)  # (batch_size,30,600)
        s_emb = self.s_emb.forward_all(sw_emb)  # (batch_size,30,1024)

        # Entity embedding
        e_emb = self.ew_emb(e)  # (batch_size,7,600)

        # Question-entity
        q_emb = self.qe_joint(q_emb, b, e_emb)  # (64,12,1024)    问题 过滤后的信息

        # Image-semantic
        v = self.vs_joint(s_emb, b, v)  # (64,30,1024)     图片 过滤后的信息

        # intra-inter Modality interaction
        # Question_Fine + Visual_Fine
        """
            Visual_Fine: "v" --(w + v)
            Question_Fine: q_emb--(q + e)
        """
        for layer in range(self.layers):
            q_emb = self.attn_AOA_fine[layer](q_emb) + q_emb

        for layer in range(self.layers):  # 通过文字模态信息，使得视觉模态学习得更好
            v = self.attn_AOA_fine[layer](v) + v
            v = self.attn_AOA_fine[layer](v, context=q_emb) + v

        # Image question

        q_emb = self.vq_joint(q_emb, b, v)  ##(64,12,1024)    # 细粒度学习 --> q_emb & v 则是 问题和图片 过滤后的信息
        ban_logits = self.classifier(q_emb.sum(1))  # ban_logits -->(64,1533) (6,3129)
        ban_probs = F.softmax(ban_logits, dim=-1)


        '''语义推理部分'''
        # logits -->(64,1533)
        # lxmert_logit -->(64, 1533)
        # ban_logits  -->(64,1533)
        """
        ban_logits.unsqueeze(1) 和 lxmert_logit.unsqueeze(1)：将两个模型的输出张量分别在第二维增加一个维度。
            [64, 1533] ---> [64, 1, 1533]
        torch.cat(..., 1)：沿着第二维将两个张量拼接起来。
            拼接后，logits 的形状将是 [64, 2, 1533]，其中2表示两个模型输出被拼接在一起。
        self.adapted_w：这是一个模型参数，用于适应性地加权不同的模型输出。它的形状是 [64, 1533]。
        torch.softmax(..., 0)：沿着第一维（即模型输出的类别维度）应用 softmax 函数，将 self.adapted_w 中的值转换为概率分布。
        这样，每个类别都会有一个权重，这些权重加起来等于1。
        """
        """
        adapted_w.unsqueeze(0)：在 adapted_w 的第一维增加一个维度，使其形状变为 [1, 64, 1533]，以匹配 logits 的形状 [64, 2, 1533]。
        torch.mul(..., ...)：将 logits 和 adapted_w.unsqueeze(0) 逐元素相乘。这一步将 adapted_w 中的权重应用到 logits 的每个类别上。
        .sum(1)：沿着第二维（即之前拼接的两个模型输出维度）对结果求和。这将 logits 的形状从 [64, 2, 1533] 减少到 [64, 1533]，得到最终的模型输出。
        """

        # prob_add = torch.cat([ban_probs.unsqueeze(1), xvlm_probs.unsqueeze(1)], 1)
        prob_add = torch.cat([ISubGVQA_logist.unsqueeze(1), xvlm_probs.unsqueeze(1), ban_probs.unsqueeze(1)], 1)
        adapted_w = torch.softmax(self.adapted_w, 0) #adapted_w -->(64,1533)
        prob_add = torch.mul(prob_add, adapted_w.unsqueeze(0)).sum(1)


        # prob_add = F.softmax(prob_add, dim=-1)
        # prob_add = ban_probs + xvlm_probs
        return prob_add,ban_probs,xvlm_probs,ISubGVQA_logist,xvlm_ce_loss


def build_ban_fusion(dataset, args, dim_1, dim_2, gamma, omega, priotize_using_counter=False):
    """
    :param dim_1: 1st modality dimension.
    :param dim_2: 2nd modality dimension.
    :param gamma: number of residual.
    :param omega: ratio fusion between two modalities.
    :return: joint representation between two modalities.
    """
    """
    # dataset: 数据集对象，包含数据集相关信息。
    # args: 参数对象，包含模型训练和结构的配置参数。
    # dim_1: 第一种模态的维度。
    # dim_2: 第二种模态的维度。
    # gamma: 残差连接的数量。   vs_joint、qe_joint == 1, vq_joint == 2
    # omega: 两种模态融合的比例。 v:0.01   q:0.1     
    # priotize_using_counter: 是否使用计数器，如果为True，则使用计数器来增强模型的表现。
    """

    b_net = []  # 初始化一个空列表，用于存储b模态的网络层。
    q_prj = []  # 初始化一个空列表，用于存储q模态的投影层。
    c_prj = []  # 初始化一个空列表，用于存储c模态（如果有的话）的投影层。
    b_att = BiAttention(dim_1, dim_2, dim_1, gamma)  # 创建双注意力模块，用于处理两种模态的交互。

    use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter    # 根据是否需要使用计数器来决定use_counter的值。

    if use_counter or priotize_using_counter:
        objects = 10  # minimum number of boxes
    for i in range(args.gamma):   # 循环gamma次，构建每个残差连接的网络结构。 #--gamma 2
        b_net.append(BCNet(dim_1, dim_2, dim_1, None, k=1))  # 添加BCNet层到b_net列表。
        q_prj.append(FCNet([dim_1, dim_1], '', .2))  # 添加全连接网络层到q_prj列表。
        if use_counter or priotize_using_counter:
            c_prj.append(FCNet([objects + 1, dim_1], 'ReLU', .0))  # 如果使用计数器，添加对应的FCNet层到c_prj列表。

    if use_counter:
        counter = Counter(objects)  # 如果使用计数器，创建计数器实例。
    else:
        counter = None  # 如果不使用计数器，设置为None。
    # 返回BanFusion类的实例，它是一个神经网络模块，用于融合两种模态的数据。
    return BanFusion(dataset, b_att, b_net, q_prj, c_prj, counter, gamma, omega)


def build_XVLM(config,args):
    model = XVLM(config=config)  # 创建模型实例
    model.load_pretrained(args.checkpoint, config, is_eval=True)  # 加载预训练模型
    device = torch.device(args.device)  # 设置设备，如果使用GPU训练，则为'cuda'
    model = model.to(device)  # 将模型发送到设备
    return model



def build_CFRF_Model(ban_dataset, args,xvlm_config):
    # lxrt_encoder = build_lxmert(dataset, args)
    XVLM_model = build_XVLM(xvlm_config,args)
    ISubGVQA_model =build_model(args)

    # Initial question embedding
    w_emb = WordEmbedding(ban_dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)

    # Initial stat-word embedding
    sw_emb = WordEmbedding(ban_dataset.dictionary.ntoken, 300, .0, args.op)
    s_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)

    # Initial entity embedding
    ew_emb = WordEmbedding(ban_dataset.dictionary.ntoken, 300, .0, args.op)

    if hasattr(args, 'tfidf'):
        if args.dataset == 'GQA':
            w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data/gqa')
            sw_emb = tfidf_loading(args.tfidf, sw_emb, args, 'data/gqa')
            ew_emb = tfidf_loading(args.tfidf, ew_emb, args, 'data/gqa')

        elif args.dataset == 'VQA':
            w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data/vqa')
            sw_emb = tfidf_loading(args.tfidf, sw_emb, args, 'data/vqa')
            ew_emb = tfidf_loading(args.tfidf, ew_emb, args, 'data/vqa')

    qe_joint = build_ban_fusion(ban_dataset, args, args.num_hid, 600, gamma=1, omega=args.omega_q)
    vs_joint = build_ban_fusion(ban_dataset, args, args.num_hid, ban_dataset.v_dim, gamma=1, omega=args.omega_v)
    vq_joint = build_ban_fusion(ban_dataset, args, args.num_hid, args.num_hid, args.gamma, omega=1, priotize_using_counter=False)
    # qe_joint = None
    # vs_joint = None
    # vq_joint = None
    # w_emb, q_emb, sw_emb, s_emb, ew_emb = None,None,None,None,None
    if args.dataset == 'GQA':
        classifier = SimpleClassifier(args.num_hid, args.num_hid * 2, ban_dataset.num_ans_candidates, args)
    elif args.dataset == 'VQA': # 可能需要修改num_hid
        classifier = SimpleClassifier(args.num_hid, args.num_hid * 2, ban_dataset.num_ans_candidates, args)

    return CFRF_Model(ban_dataset, args, XVLM_model,ISubGVQA_model, w_emb, q_emb, sw_emb, s_emb, ew_emb, qe_joint, vs_joint, vq_joint,
                        classifier, args.gamma)

