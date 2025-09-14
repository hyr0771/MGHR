import copy

from xvlm.models.xbert import BertLMHeadModel
from xvlm.models.xroberta import RobertaForCausalLM

from xvlm.models import XVLMBase, load_pretrained

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class XVLM(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False, config_text=None)

        assert isinstance(config['pad_token_id'], int)  # 检查 config 中的 pad_token_id 是否为整数类型
        self.pad_token_id = config['pad_token_id']      # 设置填充标记的 ID
        config_enc = self.text_encoder.config       # 获取文本编码器的配置

        # 设置文本层数和跨模态层数
        self.num_text_layers = config_enc.fusion_layer
        self.num_cross_layers = config_enc.num_hidden_layers - config_enc.fusion_layer
        assert config['num_dec_layers'] == self.num_cross_layers, "initialization not implemented"  # 确保解码层数与跨模态层数一致，否则报错

        config_dec = copy.deepcopy(config_enc)       # 深度复制文本编码器配置，用于解码器的配置
        config_dec.encoder_width = config_enc.hidden_size       # 设置解码器编码宽度为文本编码器的隐藏层大小
        config_dec.fusion_layer = 0  # start index          # 解码器的融合层从第0层开始
        config_dec.num_hidden_layers = config['num_dec_layers']     # 设置解码器的隐藏层数量
        # 设置跨模态和解码器编码宽度
        self.cross_encoder_width = config_enc.encoder_width  # i.e. vision_width
        self.dec_encoder_width = config_enc.hidden_size

        if config['use_roberta']:   # 如果使用 RoBERTa，则会触发一个未实现的异常
            raise NotImplementedError("bugs to fix: with roberta, the accuracy will be extreme low")
            # self.text_decoder = RobertaForCausalLM(config=config_dec)
        else:       # 否则，使用 BertLMHeadModel 作为文本解码器
            self.text_decoder = BertLMHeadModel(config=config_dec)

        if self.dec_encoder_width != self.cross_encoder_width:      # 如果解码器宽度与跨模态编码宽度不同，则初始化部分解码器参数
            self.init_params = ['text_decoder.' + n for n, _ in self.text_decoder.named_parameters()
                                if ('crossattention.self.key' in n) or ('crossattention.self.value' in n)]
        else:
            self.init_params = []        # 否则，不需要初始化特殊参数

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        if is_eval: # 如果模型处于评估模式，则加载预训练模型的权重，并且不加载文本编码器的权重
            state_dict = load_pretrained(ckpt_rpath, config, is_eval=True)

        else:   # 如果模型不处于评估模式，则加载预训练模型的权重，并指定不加载文本相关的权重
            state_dict = load_pretrained(ckpt_rpath, config, load_text=False)

            print("### Loading pretrained text encoder", flush=True)
            for key in list(state_dict.keys()):  # 遍历状态字典中的所有键
                if config['use_roberta']:   # 如果配置中指定使用RoBERTa模型
                    if 'roberta.' in key:   # 如果键名中包含'roberta.'，则替换掉并更新状态字典中的键
                        encoder_key = key.replace('roberta.', '')
                        state_dict[encoder_key] = state_dict[key]
                else:   # 如果配置中指定使用BERT模型
                    if 'bert.' in key:  # 如果键名中包含'bert.'，则替换掉并更新状态字典中的键
                        encoder_key = key.replace('bert.', '')
                        state_dict[encoder_key] = state_dict[key]

                # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                # 将文本解码器初始化为多模态编码器（即模型.text_encoder的最后6层）
                if 'text_encoder.' in key:
                    if 'layer.' in key: # 如果键名中包含'layer.'，说明是编码器层的权重
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < self.num_text_layers:    # 如果层数小于文本编码器的层数，则删除该权重，不加载
                            del state_dict[key]
                            continue

                        elif (self.dec_encoder_width != self.cross_encoder_width) and \
                                (('crossattention.self.key' in key) or ('crossattention.self.value' in key)):
                            # 如果解码器宽度与交叉编码器宽度不同，并且键名包含交叉注意力的键，则删除该权重
                            del state_dict[key]
                            continue

                        else:   # 否则，更新键名以匹配文本解码器的层数
                            decoder_layer_num = (layer_num - self.num_text_layers)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key

                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')   # 更新状态字典中的键，将文本编码器的权重映射到文本解码器
                    state_dict[decoder_key] = state_dict[key]
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)        # 加载状态字典到模型中，并且设置为非严格模式
        print('load checkpoint from %s' % ckpt_rpath)   # 打印从哪个检查点文件加载了权重
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p]) # 打印缺失的键，排除掉包含'vision_encoder'的键
        print("unexpected_keys: ", msg.unexpected_keys)  # 打印意外的键

    def forward(self, image, quesiton, answer=None, k=None, weights=None, train=True):
        image_embeds = self.vision_encoder(image)       # 使用视觉编码器对输入图像进行编码，得到图像嵌入表示
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)    # 创建图像注意力权重，大小与图像嵌入一致，全部设置为1
        
        if train:               
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            # 将答案中的填充标记 ID 替换为 -100，用于计算损失时忽略填充值
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.pad_token_id, -100)

            # 使用文本编码器编码问题输入，同时将图像嵌入作为隐藏状态输入，返回编码结果
            question_output = self.text_encoder(quesiton.input_ids,
                                                attention_mask=quesiton.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)

            # 初始化问题状态和问题注意力列表，用于存储每个问题的编码输出
            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                # 根据每个问题对应的答案数量 n，将问题编码结果和注意力添加到列表中
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [quesiton.attention_mask[b]] * n
            # 将问题状态和注意力列表堆叠为张量
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            # 使用文本解码器对答案进行解码，计算生成答案的损失
            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask=answer.attention_mask,
                                              encoder_hidden_states=question_states,
                                              encoder_attention_mask=question_atts,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none',
                                              )

            loss = weights * answer_output.loss         
            loss = loss.sum()/image.size(0)

            return loss
            
        else:   # 如果处于推理模式，仅编码问题
            question_output = self.text_encoder(quesiton.input_ids,
                                                attention_mask=quesiton.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            # 使用 rank_answer 函数来获取前 k 个最可能的答案及其概率
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, quesiton.attention_mask, 
                                                    answer.input_ids, answer.attention_mask, k) 
            return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)      # 获取问题的数量
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token        # 提取答案序列的第一个标记（bos token），并复制 num_ques 次，作为解码器的初始输入

        # 使用文本解码器对起始标记进行解码，得到初始输出
        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        # 提取初始输出中第一个标记的 logits
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:, 1]
        # logits 的形状是 (num_questions, vocab_size)； F.softmax(logits, dim=1) 的输出形状也为 (num_questions, vocab_size)，表示词汇表中每个词的概率。
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
