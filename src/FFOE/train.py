"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import time
import torch
from tqdm import tqdm

import src.utils as utils
import torch.nn as nn
import torch.distributed as dist
from src.FFOE.trainer import Trainer
# 移除对 lxrt 的依赖，使用标准 AdamW 作为替代
from torch.optim import AdamW

import _pickle as cPickle
from xvlm.models.tokenization_roberta import RobertaTokenizer
from xvlm.models.tokenization_bert import BertTokenizer
import xvlm.utils as xvlm_utils

from transformers import CLIPTokenizerFast, CLIPTokenizer

warmup_updates = 4000


def init_weights(m):
    if type(m) == nn.Linear:
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(m.weight)


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def compute_score_with_probs(probs, labels):
    labels = torch.argmax(labels, dim=1) #从one-hot编码转为普通label标签
    # 获取每个样本的类别概率
    # logits 已经是概率分布，不再需要 argmax 操作
    # 直接从 logits 中提取出每个样本对应标签的概率
    batch_size = probs.size(0)

    # 使用 gather 获取每个样本对应标签的概率
    # logits: (batch_size, num_classes), labels: (batch_size)
    probs = torch.gather(probs, dim=1, index=labels.view(-1, 1))  # 形状为 (batch_size, 1)

    # 得分等于每个样本预测正确的概率，返回的是每个样本对应标签的概率
    return probs.squeeze(1)  # squeeze 去除冗余的维度，形状变为 (batch_size)

def train(args, model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0,config_xvlm=None):
    device = args.device
    lr_default = args.lr        #1e-3 if eval_loader is not None else 7e-4      #BAN代码中
    lr_decay_step = 2
    lr_decay_rate = .25
    lr_decay_epochs = range(10, 20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 0
    grad_clip = args.clip_norm
    bert_optim = None

    utils.create_dir(output)

    # 设置混合精度训练
    scaler = None
    autocast_func = None
    if getattr(args, 'mixed_precision', False):
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            autocast_func = autocast
            print("启用混合精度训练")
        except ImportError:
            print("警告: 无法导入混合精度模块")

    if args.model == 'CFRF_Model':
        # 计算每epoch步数以 DataLoader 长度为准（兼容分布式）
        batch_per_epoch = len(train_loader)
        t_total = batch_per_epoch * args.epochs

        # 兼容 DDP 与单卡：在底层模块上做参数切分，避免 .module 访问导致的异常
        _m = getattr(model, 'module', model)
        xvlm_params = list(_m.XVLM_model.parameters())
        xvlm_param_ids = set(map(id, xvlm_params))
        base_params = [p for p in _m.parameters() if id(p) not in xvlm_param_ids]

        # 为 XVLM 与其余部分分别构建优化器（DDP安全）
        # 以 AdamW 替代原 BertAdam；warmup 由外部调度器或简化处理
        bert_optim = AdamW(xvlm_params, lr=args.xvlm_lr)
        optim = torch.optim.Adamax(base_params, lr=lr_default)

    else:
        raise BaseException("Model not found!")

    num_batches = len(train_loader)

    criterion = {}
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    criterion["BCELoss"] = torch.nn.BCELoss()
    criterion["BCEWithLogitsLoss"] = torch.nn.BCEWithLogitsLoss(reduction='sum')

    logger = utils.Logger(os.path.join(output, 'log.txt'))
    logger.write(args.__repr__())
    best_eval_score = 0

    # utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    trainer = Trainer(args, model, criterion, optim, bert_optim)
    update_freq = int(args.update_freq)
    wall_time_start = time.time()

    if config_xvlm['use_roberta']:   # 根据配置决定使用BertTokenizer还是RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(config_xvlm['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config_xvlm['text_encoder'])
    answer_list = [answer + config_xvlm['eos'] for answer in train_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)  # 将answer_list进行分词处理，准备用于模型预测

    ISubGVQA_tokenizer = CLIPTokenizerFast.from_pretrained(args.clip_url, use_fast=True)

    # 梯度累积相关
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    effective_batch_size = args.batch_size * gradient_accumulation_steps
    logger.write(f'有效批处理大小: {effective_batch_size} (批处理大小: {args.batch_size} x 累积步数: {gradient_accumulation_steps})')

    for epoch in range(s_epoch, num_epochs):
        # 分布式训练时，需为分布式采样器设置当前 epoch，确保各 rank 采样一致性
        try:
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
        except Exception:
            pass
        total_loss = 0
        train_score = 0
        train_question_type_score = 0
        total_norm = 0
        count_norm = 0
        num_updates = 0
        t = time.time()
        if args.model == 'CFRF_Model':
            if epoch < len(gradual_warmup_steps):
                trainer.optimizer.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
                logger.write('gradual warmup lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
            elif epoch in lr_decay_epochs:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay_rate
                logger.write('decreased lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
            else:
                logger.write('lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
        else:
            raise BaseException("Model not found!")

        total_batches = len(train_loader)
        # 仅主进程显示进度条
        pbar = tqdm(train_loader, total=total_batches, desc="Processing", disable=not xvlm_utils.is_main_process())
        # for i, (a,image,question,scene_graph) in enumerate(pbar):
        for i, (v, b, w, e, q, a, ans, image, question, question_id, scene_graph) in enumerate(pbar):
            if getattr(args, 'debug_log_steps', False):
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        print(f"[rank={dist.get_rank()}] step={i}/{total_batches}")
                except Exception:
                    pass
            pbar.set_description(f"Processing batch {i}/{total_batches}")
            v = v.to(device) #(batch_size,50,2480)
            b = b.to(device) #(batch_size,50,6)
            e = e.to(device) #(batch_size,7)
            w = w.to(device) #(batch_size,30)
            q = q.to(device) #(batch_size,12)
            a = a.to(device) #(batch_size,1533)
            # ans = ans.to(device) #(batch_size,2)
            image = image.to(device)      #[batch,3,480,480]

            scene_graph.x = scene_graph.x.to(device)
            scene_graph.edge_index = scene_graph.edge_index.to(device)
            scene_graph.edge_attr = scene_graph.edge_attr.to(device)
            scene_graph.added_sym_edge = scene_graph.added_sym_edge.to(device)
            scene_graph.x_bbox = scene_graph.x_bbox.to(device)
            scene_graph.batch = scene_graph.batch.to(device)
            scene_graph.ptr = scene_graph.ptr.to(device)
            # scene_graph DataBatch(x=[142, 4], edge_index=[2, 685], edge_attr=[685], added_sym_edge=[48], x_bbox=[142, 4],batch=[142], ptr=[9])


            question_xvlm = train_loader.dataset.tokenizer(question, padding='longest', return_tensors="pt").to(device) #是一个字典   {key1:[batch,10]}
            question_ISubGVQA = ISubGVQA_tokenizer(question,return_tensors="pt", padding=True).to(device)
            question_ISubGVQA["input_ids"] = question_ISubGVQA["input_ids"].to(device)
            question_ISubGVQA["attention_mask"] = question_ISubGVQA["attention_mask"].to(device)



            # sample = [a,image, question_xvlm, answer_input, scene_graph, question_ISubGVQA]
            sample = [v, b, w, e, q, a, ans, image, question_xvlm, answer_input, scene_graph,question_ISubGVQA]

            if i < num_batches - 1 and (i + 1) % update_freq > 0:
                trainer.train_step(sample, update_params=False)
            else:
                loss, grad_norm, batch_score, batch_question_type_score = trainer.train_step(sample, update_params=True)
                total_norm += grad_norm
                count_norm += 1
                total_loss += loss.item()
                train_score += batch_score
                num_updates += 1
                if xvlm_utils.is_main_process() and num_updates % int(args.print_interval / update_freq) == 0:
                    # print("Iter: {}, Loss {:.8f}, Norm: {:.8f}, Total norm: {:.8f}, Num updates: {}, Wall time: {:.2f},"
                    #       "ETA: {} batch_score {:.4f}".format(i + 1, total_loss / ((num_updates + 1)), grad_norm, total_norm, num_updates,
                    #                        time.time() - wall_time_start, utils.time_since(t, i / num_batches), train_score/ ((num_updates + 1)) ))
                    logger.write("Iter: {}, Loss {:.8f}, Norm: {:.8f}, Total norm: {:.8f}, Num updates: {}, Wall time: {:.2f},"
                          "ETA: {} batch_score {:.4f}".format(i + 1, total_loss / ((num_updates + 1)), grad_norm, total_norm, num_updates,
                                           time.time() - wall_time_start, utils.time_since(t, i / num_batches) ,train_score / ((num_updates + 1)) ))
                    if args.testing:
                        break

                # 可选：每个 step 末尾做一次 barrier（仅用于定位不同步问题）
                if getattr(args, 'barrier_each_step', False):
                    try:
                        import torch.distributed as dist
                        if dist.is_available() and dist.is_initialized():
                            dist.barrier()
                    except Exception:
                        pass

        total_loss /= num_updates
        train_score = 100 * train_score / (num_updates * args.batch_size)
        train_question_type_score = 100 * train_question_type_score / (num_updates * args.batch_size)

        # Save per epoch (先保存模型)
        if xvlm_utils.is_main_process() and epoch >= saving_epoch:
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, trainer.optimizer)

        if eval_loader is not None:
            print("Evaluating...")
            trainer.model.train(False)
            # cfrf_score, fg_score, coarse_score, ens_score, upper_bound
            eval_cfrf_score, fg_score, coarse_score, ens_score,mid_score, bound = evaluate(model, eval_loader, args, config_xvlm)
            trainer.model.train(True)

        if xvlm_utils.is_main_process():
            logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
            logger.write('\ttrain_loss: %.8f, norm: %.8f, score: %.8f, question type score: %.8f' %
                         (total_loss, total_norm/count_norm, train_score, train_question_type_score))
        if eval_loader is not None:
            if xvlm_utils.is_main_process():
                logger.write('\tCFRF score: %.8f (%.8f)' % (100 * eval_cfrf_score, 100 * bound))
                logger.write('\tfg_score: %.8f ' % (100 * fg_score))
                logger.write('\tcoarse_score: %.8f ' % (100 * coarse_score))
                logger.write('\tens_score: %.8f ' % (100 * ens_score))
                logger.write('\tmid_score: %.8f ' % (100 * mid_score))

        # Save best epoch (验证后保存最佳模型)
        if xvlm_utils.is_main_process() and eval_loader is not None and eval_cfrf_score > best_eval_score:
            model_path = os.path.join(output, 'model_epoch_best.pth')
            utils.save_model(model_path, model, epoch, trainer.optimizer)
            best_eval_score = eval_cfrf_score

def evaluate(model, dataloader, args, config_xvlm):
    device = args.device
    cfrf_score = 0
    ens_score = 0
    fg_score = 0
    coarse_score = 0
    upper_bound = 0
    num_data = 0
    mid_score = 0
    ISubGVQA_tokenizer = CLIPTokenizerFast.from_pretrained(args.clip_url, use_fast=True)
    if config_xvlm['use_roberta']:   # 根据配置决定使用BertTokenizer还是RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(config_xvlm['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config_xvlm['text_encoder'])
    answer_list = [answer + config_xvlm['eos'] for answer in dataloader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)  # 将answer_list进行分词处理，准备用于模型预测

    with torch.no_grad():
        total_batches = len(dataloader)
        pbar = tqdm(dataloader, total=total_batches, desc="Processing", disable=not xvlm_utils.is_main_process())
        #   target,image,question_xvlm,scene_graph
        # for i, (a,image,question,scene_graph) in enumerate(pbar):
        for i, (v, b, w, e, q, a, ans, image, question, question_id, scene_graph) in enumerate(pbar):
            pbar.set_description(f"Processing batch {i}/{total_batches}")
            v = v.to(device)  # (batch_size,50,2480)
            b = b.to(device)  # (batch_size,50,6)
            e = e.to(device)  # (batch_size,7)
            w = w.to(device)  # (batch_size,30)
            q = q.to(device)  # (batch_size,12)
            a = a.to(device)  # (batch_size,1533)
            # ans = ans.to(device)  # (batch_size,2)
            image = image.to(device)  # [batch,3,480,480]
            scene_graph.x = scene_graph.x.to(device)
            scene_graph.edge_index = scene_graph.edge_index.to(device)
            scene_graph.edge_attr = scene_graph.edge_attr.to(device)
            scene_graph.added_sym_edge = scene_graph.added_sym_edge.to(device)
            scene_graph.x_bbox = scene_graph.x_bbox.to(device)
            scene_graph.batch = scene_graph.batch.to(device)
            scene_graph.ptr = scene_graph.ptr.to(device)

            question_xvlm = dataloader.dataset.tokenizer(question, padding='longest', return_tensors="pt").to(device)  # 是一个字典   {key1:[batch,10]}
            question_ISubGVQA = ISubGVQA_tokenizer(question, return_tensors="pt", padding=True).to(device)
            question_ISubGVQA["input_ids"] = question_ISubGVQA["input_ids"].to(device)
            question_ISubGVQA["attention_mask"] = question_ISubGVQA["attention_mask"].to(device)


            if args.model == 'CFRF_Model':
                #[v, b, w, e, q, a, ans,image, question_xvlm, question_id, answer_input, scene_graph,question_ISubGVQA]
                # fusion_preds, ban_preds, xvlm_preds = model(v, b, w, e, q ,image, question_xvlm["input_ids"],
                #                                             question_xvlm["token_type_ids"],question_xvlm["attention_mask"],answer_input)
                result = model(v, b, w, e, q, image, question_xvlm, answer_input,scene_graph,question_ISubGVQA)
                if len(result) == 5:  # 训练模式，返回5个值
                    fusion_preds, ban_preds, xvlm_preds, ISubGVQA_preds, xvlm_ce_loss = result
                else:  # 验证模式，返回4个值
                    fusion_preds, ban_preds, xvlm_preds, ISubGVQA_preds = result

                ens_preds = fusion_preds + ban_preds + xvlm_preds + ISubGVQA_preds
                fg_score += compute_score_with_logits(ban_preds, a).sum()
                coarse_score += compute_score_with_logits(xvlm_preds, a).sum()
                ens_score += compute_score_with_logits(ens_preds, a).sum()
                mid_score += compute_score_with_logits(ISubGVQA_preds, a).sum()
                final_preds = fusion_preds

            batch_score = compute_score_with_logits(final_preds, a).sum()
            cfrf_score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += final_preds.size(0)

    # 分布式聚合（求和），再用全局 num_data 归一化
    if xvlm_utils.is_dist_avail_and_initialized():
        import torch.distributed as dist
        device = args.device if isinstance(args.device, torch.device) else torch.device(args.device)
        tensor = torch.tensor([cfrf_score, fg_score, coarse_score, ens_score, mid_score, upper_bound, float(num_data)], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        cfrf_score, fg_score, coarse_score, ens_score, mid_score, upper_bound, num_data = tensor.tolist()

    # 防止除零
    denom = max(float(num_data), 1.0)
    cfrf_score = cfrf_score / denom
    fg_score = fg_score / denom
    coarse_score = coarse_score / denom
    ens_score = ens_score / denom
    mid_score = mid_score / denom
    upper_bound = upper_bound / denom

    return cfrf_score, fg_score, coarse_score, ens_score, mid_score, upper_bound


def tensor_to_list(var):
    """
    如果变量是PyTorch张量，则将其转换为列表。

    参数:
        var: 要检查的变量。

    返回:
        如果var是PyTorch张量，返回转换后的列表；
        否则，返回原始变量。
    """
    if isinstance(var, torch.Tensor):
        return var.tolist()
    else:
        return var

def predict(model, dataloader, args, config_xvlm):
    device = args.device
    cfrf_quesid2ans = {}
    ens_quesid2ans = {}
    fg_quesid2ans = {}
    cg_quesid2ans = {}
    mid_quesid2ans = {}
    #XVLM
    if config_xvlm['use_roberta']:   # 根据配置决定使用BertTokenizer还是RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(config_xvlm['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config_xvlm['text_encoder'])
    answer_list = [answer + config_xvlm['eos'] for answer in dataloader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)  # 将answer_list进行分词处理，准备用于模型预测

    ISubGVQA_tokenizer = CLIPTokenizerFast.from_pretrained(args.clip_url, use_fast=True)

    if args.dataset == 'GQA':
        label2ans_path = "./data/gqa/cache/trainval_label2ans.pkl"
        label2ans = cPickle.load(open(label2ans_path, 'rb'))
    elif args.dataset == 'VQA':
        label2ans_path = "./data/vqa/cache/trainval_label2ans.pkl"
        label2ans = cPickle.load(open(label2ans_path, 'rb'))

    with torch.no_grad():
        # features, spatials, stat_features, entity, question_ban, question_id, image, question_xvlm    #dataset:__getitem__()
        total_batches = len(dataloader)
        pbar = tqdm(dataloader, total=total_batches, desc="Processing", disable=not xvlm_utils.is_main_process())
        for i, (v, b, w, e, q, image, question, question_id, scene_graph) in enumerate(pbar):
            pbar.set_description(f"Processing batch {i}/{total_batches}")
            v = v.to(device)  # (batch_size,50,2480)
            b = b.to(device)  # (batch_size,50,6)
            e = e.to(device)  # (batch_size,7)
            w = w.to(device)  # (batch_size,30)
            q = q.to(device)  # (batch_size,12)
            image = image.to(device)  # [batch,3,480,480]
            scene_graph.x = scene_graph.x.to(device)
            scene_graph.edge_index = scene_graph.edge_index.to(device)
            scene_graph.edge_attr = scene_graph.edge_attr.to(device)
            scene_graph.added_sym_edge = scene_graph.added_sym_edge.to(device)
            scene_graph.x_bbox = scene_graph.x_bbox.to(device)
            scene_graph.batch = scene_graph.batch.to(device)
            scene_graph.ptr = scene_graph.ptr.to(device)

            question_xvlm = dataloader.dataset.tokenizer(question, padding='longest', return_tensors="pt").to(
                device)  # 是一个字典   {key1:[batch,10]}
            question_ISubGVQA = ISubGVQA_tokenizer(question, return_tensors="pt", padding=True).to(device)
            question_ISubGVQA["input_ids"] = question_ISubGVQA["input_ids"].to(device)
            question_ISubGVQA["attention_mask"] = question_ISubGVQA["attention_mask"].to(device)
            # final_preds = None

            if args.model == 'CFRF_Model':
                result = model(v, b, w, e, q, image, question_xvlm, answer_input,scene_graph,question_ISubGVQA)
                if len(result) == 5:  # 训练模式，返回5个值
                    fusion_preds, ban_preds, xvlm_preds, ISubGVQA_preds, xvlm_ce_loss = result
                else:  # 验证模式，返回4个值
                    fusion_preds, ban_preds, xvlm_preds, ISubGVQA_preds = result


                ens_preds = fusion_preds + ban_preds + xvlm_preds + ISubGVQA_preds

                fusion_predicts = torch.max(fusion_preds, 1)[1].data  # argmax
                ens_predicts = torch.max(ens_preds, 1)[1].data  # argmax
                ban_predicts  = torch.max(ban_preds, 1)[1].data  # argmax
                xvlm_predicts = torch.max(xvlm_preds, 1)[1].data  # argmax
                ISubGVQA_predicts = torch.max(ISubGVQA_preds, 1)[1].data  # argmax

                question_id = tensor_to_list(question_id)
                ens_predicts = tensor_to_list(ens_predicts)
                fusion_predicts = tensor_to_list(fusion_predicts)
                ban_predicts = tensor_to_list(ban_predicts)

                for qid, ens_predict, fusion_predict, ban_predict, xvlm_predict,ISubGVQA_predict in zip(question_id,
                                                                                 list(ens_predicts),
                                                                                 list(fusion_predicts),
                                                                                 list(ban_predicts),
                                                                                 list(xvlm_predicts),
                                                                                 list(ISubGVQA_predicts)):
                    ens_quesid2ans[qid] = label2ans[ens_predict]
                    cfrf_quesid2ans[qid] = label2ans[fusion_predict]
                    fg_quesid2ans[qid] = label2ans[ban_predict]
                    cg_quesid2ans[qid] = label2ans[xvlm_predict]
                    mid_quesid2ans[qid] = label2ans[ISubGVQA_predict]


            else:
                raise BaseException("Model not found!")

    return cfrf_quesid2ans, fg_quesid2ans, ens_quesid2ans ,cg_quesid2ans, mid_quesid2ans