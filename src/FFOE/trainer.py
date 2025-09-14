"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import logging

import torch
import torch.distributed as dist
import src.utils as utils
import contextlib
from collections import defaultdict, OrderedDict
from src.meters import AverageMeter, TimeMeter

class Trainer(object):
    """
    Main class for training.
    """
    def __init__(self, args, model, criterion, optimizer=None, bert_optimizer=None):
        self.args = args

        # copy model and criterion on current device
        self.model = model.to(self.args.device)
        # self.criterion = criterion.to(self.args.device)
        self.criterion_BCEWithLogitsLoss = criterion["BCEWithLogitsLoss"].to(self.args.device)
        self.BCELoss = criterion["BCELoss"].to(self.args.device)

        # initialize meters
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        self.meters['wall'] = TimeMeter()      # wall time in seconds

        self._buffered_stats = defaultdict(lambda: [])
        self._flat_grads = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        if optimizer is not None:
            self._optimizer = optimizer
        self._bert_optim = bert_optimizer
        self.total_loss = 0.0
        self.train_score = 0.0
        self.total_norm = 0.0
        self.count_norm = 0.0

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    def bert_optimizer(self):
        if self._bert_optim is None:
            self._build_optimizer()
        return self._bert_optim

    def _build_optimizer(self):
        # self._optimizer = optim.build_optimizer(self.args, self.model.parameters())
        # self._optimizer =
        # self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self._optimizer)
        pass

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        # if distributed_utils.is_master(self.args):  # only save one checkpoint
        #     extra_state['train_meters'] = self.meters
        #     utils.save_state(
        #         filename, self.args, self.model, self.criterion, self.optimizer,
        #         self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
        #     )
        pass

    def load_checkpoint(self, filename):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_model_state(filename, self.model)

        if last_optim_state is not None:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            if last_optim['criterion_name'] == self.criterion.__class__.__name__:
                # self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
                if last_optim['optimizer_name'] == self.optimizer.__class__.__name__:
                    self.optimizer.load_state_dict(last_optim_state)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters = extra_state['train_meters']
            del extra_state['train_meters']

        return extra_state

    def train_step(self, sample, update_params=True):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        # seed = self.args.seed + self.get_num_updates()
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)

        # forward and backward pass
        sample = self._prepare_sample(sample)       # 准备输入数据 sample，例如将其移到 GPU 上。
        loss, sample_size, oom_fwd, batch_score, batch_question_type_score = self._forward(sample)  # 执行前向传播，计算损失等。

        # 分布式场景下，如任一 rank 在前向期间 OOM/返回空损失，则所有 rank 同步跳过本 step，避免集体通信错序
        if getattr(self.args, 'distributed', False) and dist.is_available() and dist.is_initialized():
            oom_flag_tensor = torch.tensor(
                1 if (oom_fwd == 1 or loss is None) else 0,
                device=self.args.device,
                dtype=torch.int32,
            )
            dist.all_reduce(oom_flag_tensor, op=dist.ReduceOp.SUM)
            if int(oom_flag_tensor.item()) > 0:
                self.zero_grad()
                # 统一返回，保持各 rank 步调一致
                return None

        oom_bwd = self._backward(loss)   # 执行反向传播，计算梯度。

        # 分布式场景下，如任一 rank 在反向期间 OOM，则所有 rank 同步跳过，避免后续 allreduce 错序
        if getattr(self.args, 'distributed', False) and dist.is_available() and dist.is_initialized():
            oom_bwd_tensor = torch.tensor(
                1 if (oom_bwd == 1) else 0,
                device=self.args.device,
                dtype=torch.int32,
            )
            dist.all_reduce(oom_bwd_tensor, op=dist.ReduceOp.SUM)
            if int(oom_bwd_tensor.item()) > 0:
                self.zero_grad()
                return None

        # buffer stats and logging outputs
        # self._buffered_stats['sample_sizes'].append(sample_size)
        self._buffered_stats['sample_sizes'].append(1)      # 将样本数量（此处为 1）加入缓存统计数据。
        self._buffered_stats['ooms_fwd'].append(oom_fwd)  # 记录前向传播过程中发生的 out-of-memory（OOM）错误数量。
        self._buffered_stats['ooms_bwd'].append(oom_bwd)  # 记录反向传播过程中发生的 OOM 错误数量。

        # update parameters
        if update_params:
            # gather logging outputs from all replicas
            sample_sizes = self._buffered_stats['sample_sizes']  # 获取缓存中保存的样本数量。
            ooms_fwd = self._buffered_stats['ooms_fwd']  # 获取前向传播的 OOM 错误次数。
            ooms_bwd = self._buffered_stats['ooms_bwd']  # 获取反向传播的 OOM 错误次数。
            ooms_fwd = sum(ooms_fwd)  # 计算前向传播中 OOM 错误的总次数。
            ooms_bwd = sum(ooms_bwd)  # 计算反向传播中 OOM 错误的总次数。

            # aggregate stats and logging outputs
            # 统计并汇总日志输出
            grad_denom = sum(sample_sizes)  # 计算用于梯度缩放的分母，通常是样本总数。

            grad_norm = 0
            try:
                # all-reduce and rescale gradients, then take an optimization step
                grad_norm = self._all_reduce_and_rescale(grad_denom)  # 聚合并重新缩放梯度。
                self._opt()  # 执行优化器更新。

                # update meters
                if grad_norm is not None:
                    self.meters['gnorm'].update(grad_norm)  # 更新梯度范数指标。
                    self.meters['clip'].update(1. if grad_norm > self.args.clip_norm else 0.)  # 记录是否发生梯度裁剪。

                self.meters['oom'].update(ooms_fwd + ooms_bwd)  # 更新 OOM 错误的总次数。

            except OverflowError as e:
                self.zero_grad()  # 将梯度归零，防止梯度爆炸。
                print('| WARNING: overflow detected, ' + str(e))

            self.clear_buffered_stats() # 清除缓存的统计数据。

            return loss, grad_norm, batch_score, batch_question_type_score
        else:
            return None  # buffering updates

    def _forward(self, sample, eval=False):
        # prepare model and optimizer
        if eval:
            self.model.eval()
        else:
            self.model.train()
        loss = None
        # sample_size = 0
        oom = 0
        batch_score = 0
        batch_question_type_score = None
        if sample is not None:
            try:
                with torch.no_grad() if eval else contextlib.ExitStack():
                    # calculate loss and sample size
                    answers = sample[5]
                    # teacher_logits = sample[-1]

                    if self.args.model == 'CFRF_Model':
                        #[w, q, a, attr, e, ans, v, b, s]   原先
                        # [v, b, w, e, q, a, ans, image7, question_xvlm, question_id, answer_input, scene_graph, question_ISubGVQA]
                        #[a,image, question_xvlm, answer_input,scene_graph,question_ISubGVQA]
                            #image, question_xvlm, answer_input, scene_graphs, question_ISubGVQA
                        # v b q s e w 原先model中需要的参数
                        # print("^^^^^^^^^^^^^^^^^^^^^answer_probs^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                        # print(sample[7].shape) torch.Size([32, 3, 480, 480])
                        # print(sample[8]["input_ids"].shape) torch.Size([32, 10])
                        # print(sample[8]["token_type_ids"].shape) torch.Size([32, 10])
                        # print(sample[8]["attention_mask"].shape) torch.Size([32, 10])
                        # print("^^^^^^^^^^^^^^^^^^^^^answer_probs^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                        #sample = [a,image1, question_xvlm, answer_input, scene_graph, question_ISubGVQA]
                        # prob_add, ban_preds, xvlm_preds = self.model(sample[1], sample[2],sample[3],sample[4],sample[5])
                        # sample = [v, b, w, e, q, a, ans, image7, question_xvlm8, answer_input9, scene_graph10,question_ISubGVQA11]

                        prob_add, ban_preds, xvlm_preds, ISubGVQA_preds, xvlm_ce_loss = self.model(
                            sample[0], sample[1], sample[2], sample[3], sample[4],
                            sample[7], sample[8], sample[9], sample[10], sample[11],
                            answers if getattr(self.args, 'tune_xvlm', False) else None,
                        )



                        # print("^^^^^^^^^^^^^^^^^^^^^begin^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                        # print("ban_preds.shape")
                        # print(ban_preds.shape)  #[8, 3129]
                        # print(ban_preds)
                        # print("answers.shape")
                        # print(answers.shape)        #[8, 3129]        [0,0,0,0,0,1,......] [0,0,0.3,...0.6...,1,....]
                        # print("^^^^^^^^^^^^^^^^^^^^^end^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


                        # Fusion loss
                        # logging.debug(f'trainer.py  answers.shape:{answers.shape}')
                        # answers_label = torch.argmax(answers, dim=1) #从one-hot编码转为普通label标签
                        # print(f"prob_add.shape：：{prob_add.shape}")
                        loss_1 = self.BCELoss(prob_add.float(), answers)
                        loss_1 /= answers.size()[0]

                        # Ban loss
                        loss_2 = self.BCELoss(ban_preds.float(), answers)
                        loss_2 /= answers.size()[0]

                        # XVLM 微调（方案B）：使用 XVLM 原生 decoder CE（由 base_model 返回 xvlm_ce_loss）
                        loss_3 = xvlm_ce_loss if (xvlm_ce_loss is not None) else torch.tensor(0.0, device=self.args.device)

                        # ISubGVQA loss
                        loss_4 = self.BCELoss(ISubGVQA_preds.float(), answers)
                        loss_4 /= answers.size()[0]

                        # 如果需要将 XVLM 的 CE 引入总损失，可通过 self.args.tune_xvlm 控制
                        if getattr(self.args, 'tune_xvlm', False):
                            lambda_xvlm = getattr(self.args, 'lambda_xvlm', 0.3)
                            loss = self.args.fusion_ratio * loss_1 + loss_2 + loss_4 + lambda_xvlm * loss_3
                        else:
                            loss = self.args.fusion_ratio * loss_1 + loss_2 + loss_4
                        # loss = 1.0*loss_1 + 0.31*loss_2 + 0.23*loss_3 + 0.46*loss_4     #DS
                        # loss = self.args.fusion_ratio * loss_1 + loss_2
                        # fusion_preds = ban_preds
                        final_preds = prob_add
                    batch_score = compute_score_with_logits(final_preds, answers).sum()
                    # logging.debug(f'trainer.py  batch_score:{batch_score}')

            except RuntimeError as e:
                if not eval and 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = 1
                    loss = None
                else:
                    raise e
        return loss, len(sample[0]), oom, batch_score, batch_question_type_score  # TODO: Not sure about sample size, need to recheck

    def _backward(self, loss):
        oom = 0
        if loss is not None:
            try:
                # backward pass
                # loss.backward()
                loss.backward(retain_graph=False)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = 1
                    self.zero_grad()
                else:
                    raise e
        return oom

    def _all_reduce_and_rescale(self, grad_denom):
        # flatten grads into a single buffer and all-reduce
        flat_grads = self._flat_grads = self._get_flat_grads(self._flat_grads)

        # rescale and clip gradients
        flat_grads.div_(grad_denom)
        grad_norm = utils.clip_grad_norm_(flat_grads, self.args.clip_norm)

        # copy grads back into model parameters
        self._set_flat_grads(flat_grads)

        return grad_norm

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                continue
                #raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         #'Use the param in the forward pass or set requires_grad=False')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset+numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]

    def _set_flat_grads(self, new_grads):
        grads = self._get_grads()
        offset = 0
        for g in grads:
            numel = g.numel()
            g.copy_(new_grads[offset:offset+numel].view_as(g))
            offset += numel

    def _opt(self):
        # take an optimization step
        self.optimizer.step()
        if self._bert_optim is not None:
            self._bert_optim.step()
        self.zero_grad()
        self._num_updates += 1

        # update learning rate
        # self.lr_scheduler.step_update(self._num_updates)

    def zero_grad(self):
        self.optimizer.zero_grad()
        if self._bert_optim is not None:
            self._bert_optim.zero_grad()

    def clear_buffered_stats(self):
        self._buffered_stats.clear()

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        return utils.move_to_cuda(sample)

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, update_params=False)
        self.zero_grad()
        self.clear_buffered_stats()

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

