"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from src.FFOE.dataset import Dictionary, GQAFeatureDataset ,VQAFeatureDataset
import src.FFOE.base_model as base_model
from src.FFOE.train import train,evaluate
import src.utils as utils

# 导入优化的数据处理模块
from src.FFOE.optimized_collate import (
    compatible_collate_fn,
    monitored_compatible_collate_fn
)
import random
import numpy as np
import gc
import psutil

import ruamel.yaml as yaml
from xvlm.dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from PIL import Image
from torchvision import transforms
import xvlm.utils as xvlm_utils
# from xvlm.utils.checkpointer import Checkpointer
# from xvlm.utils.hdfs_io import hmkdir, hexists
try:
    import _pickle as pickle
except:
    import pickle

torch._dynamo.config.cache_size_limit = 1024  # 默认值较小，增大到1024

def memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE CFRF HYPER-PARAMETERS------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    # parser.add_argument('--output', type=str, default='saved_models/GQA',
    #                     help='save file directory')
    parser.add_argument('--output', type=str, default='saved_models/GQA/replicate_result_default',
                        help='save file directory')


    # Utilities
    parser.add_argument('--seed', type=int, default=1204,help='random seed')
    parser.add_argument('--epochs', type=int, default=20,help='the number of epoches')
    parser.add_argument('--lr', default=2.8e-4, type=float, metavar='lr',help='initial learning rate')


    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=32,help='batch size')
    parser.add_argument('--update_freq', default='4', metavar='N',help='update parameters every n batches in an epoch')


    # Data
    parser.add_argument('--use_both', action='store_true',help='use both train/val datasets to train?')


    # Choices of models
    parser.add_argument('--model', type=str, default='CFRF_Model', choices=['CFRF_Model'],
                        help='the model we use')
    parser.add_argument('--dataset', type=str, default='GQA', choices=['GQA','VQA'],
                        help='Dataset to train and evaluate')

    # INTERACTION LEARNING COMPONENTS HYPER-PARAMETERS------------------------------------------------------------------
    # BAN
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')
    parser.add_argument('--counter_act', type=str, default='zhang', choices=['zhang'],
                        help='the counter activation')


    # CONSTANT HYPER-PARAMETERS (Advanced hyper-params for testing, experimenting or fine-tuning)------------------------
    # Utilities - support testing, gpu training or sampling
    parser.add_argument('--testing', action='store_true', default=False,
                        help='for fast testing 1 epoch')
    parser.add_argument('--print_interval', default=200, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')
    parser.add_argument('--weight_init', type=str, default='none', choices=['none', 'kaiming_normal'],
                        help='dynamic weighting with Kaiming normalization')

    # Bounding box set
    parser.add_argument('--max_boxes', default=50, type=int, metavar='N',
                        help='number of maximum bounding boxes for K-adaptive')
    # Stat word
    parser.add_argument('--num_stat_word', default=30, type=int, metavar='N',
                        help='number of statistical word')

    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Framework hyper-params
    parser.add_argument('--activation', type=str, default='swish', choices=['relu', 'swish'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.45, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Debugging
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    # 移除未使用的 LXMERT 相关参数


    # Optimization
    # 移除未使用的 mceLoss
    # parser.add_argument('--lxmert_lr', default=5e-5, type=float, metavar='lr',
    #                     help='initial learning rate')
    parser.add_argument('--xvlm_lr', default=5e-5, type=float, metavar='lr',
                        help='initial learning rate')


    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    # parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    # 移除与 LXMERT 配置相关的多余层数参数

    # LXMERT Pre-training Config
    # 移除 LXMERT 预训练相关任务开关

    # Training configuration
    # 移除旧的 multiGPU 标志（改用 --distributed + torchrun）
    # parser.add_argument("--numWorkers", dest='num_workers', default=8)

    # Fine-tuning arguments
    parser.add_argument('--omega_q', type=float, default=0.1,
                        help='omega for control the effect of question instructions')
    parser.add_argument('--omega_v', type=float, default=0.01,
                        help='omega for control the effect of image semantics')
    parser.add_argument('--fusion_ratio', type=float, default=0.1,
                        help='ratio for control the effect of adapted weight')

    parser.add_argument('--topk', default='6', type=int)

    # Modal interaction hyperparameters
    parser.add_argument('--AOA_layers', type=int, default=6,
                        help='The number of layers of the AOA')

    #XVLM
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./xvlm/configs/VQA_480.yaml')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--xvlm_seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--evaluate', action='store_true')

    #ISubGVQA
    parser.add_argument("--general_hidden_dim", type=int, default=300)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--use_all_instrs", action="store_true", default=False)
    parser.add_argument("--use_global_mask", action="store_true", default=False)
    parser.add_argument("--node_classification", action="store_true", default=False)
    parser.add_argument("--sampler_type", type=str,default="imle")
    parser.add_argument("--sample_k", type=int, default=2)
    parser.add_argument("--nb_samples", default=1, type=int, metavar="N")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--beta", default=10.0, type=float)
    parser.add_argument("--tau", default=1.0, type=float)

    parser.add_argument("--gnn_gating", type=int, default=1)
    parser.add_argument("--use_instruction", type=int, default=1)
    parser.add_argument("--use_masking", type=int, default=1)
    parser.add_argument("--use_mgat", type=int, default=0)
    parser.add_argument("--mgat_masks", nargs="+", type=float, default=[1.0, 1.0, 1.0, 0.15])

    parser.add_argument("--use_topk", type=int, default=True)
    parser.add_argument("--interpretable_mode", type=int, default=False)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--concat_instr", type=int, default=0)
    parser.add_argument("--embed_cat", type=int, default=0)
    parser.add_argument("--text_sampling", action="store_true", default=False)
    parser.add_argument("--mgat_layers", default=4, type=int, metavar="N")
    parser.add_argument("--clip_url", default='/home/hjm/.cache/huggingface/transformers/openai/clip-vit-base-patch32/')
    
    # 内存优化参数
    # 精简：多卡下默认关闭动态批/混合精度/内存高效注意力等参数，可按需恢复
    parser.add_argument('--dry-run', action='store_true', help='仅加载模块，不执行训练')
    # XVLM 微调/蒸馏相关
    parser.add_argument('--tune_xvlm', action='store_true', default=False,
                        help='微调 XVLM：训练中对 XVLM 概率计算保留梯度，并引入 XVLM 的 BCE 分类损失')
    parser.add_argument('--lambda_xvlm', type=float, default=0.3,
                        help='XVLM BCE 损失在总损失中的权重（仅当 --tune_xvlm 时生效）')

    # 调试/稳定性开关（DDP 故障排查）
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='启用混合精度训练（AMP），降低显存占用、提升训练速度')
    parser.add_argument('--disable_syncbn', action='store_true', default=False,
                        help='禁用 SyncBatchNorm（用 BatchNorm1d 替代），用于排查多卡卡死/不同步问题')
    parser.add_argument('--dataloader_single_worker', action='store_true', default=False,
                        help='将 DataLoader 设为单 worker（num_workers=0, persistent_workers=False）以排查数据加载问题')
    parser.add_argument('--train_drop_last', action='store_true', default=False,
                        help='训练 DataLoader 丢弃最后不完整 batch（多卡建议开启，避免不同步）')
    parser.add_argument('--debug_log_steps', action='store_true', default=False,
                        help='打印每个 rank 的 step 对齐日志，帮助定位卡住步数')
    parser.add_argument('--barrier_each_step', action='store_true', default=False,
                        help='每个训练 step 末尾加一次 dist.barrier()（仅调试用，会显著降速）')

    # DDP 性能/收敛开关
    parser.add_argument('--strict_ddp', action='store_true', default=False,
                        help='启用严格 DDP（find_unused_parameters=False），要求所有参数每步都参与计算')
    parser.add_argument('--ddp_bucket_cap_mb', type=int, default=100,
                        help='DDP 梯度桶大小（MB），更大可减少 allreduce 次数，默认100MB')

    # Return args
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # （已移除）外部优化配置加载与应用，DDP 训练不依赖该机制
    
    #XVLM
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    xvlm_utils.init_distributed_mode(args)  # 初始化分布式训练模式（会在分布式时设置 args.gpu 并绑定设备）
    # 确保日志及时刷新（多进程下便于观察 rank 输出）
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    
    # 如通过 CLI 请求禁用 SyncBatchNorm，则设置环境变量，底层场景图模块会读取
    if getattr(args, 'disable_syncbn', False):
        os.environ['DISABLE_SYNCBN'] = '1'
    # 分布式时按本地rank绑定设备；否则按用户指定的 device
    if getattr(args, 'distributed', False):
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device(args.device)
    args.device = device
    world_size = xvlm_utils.get_world_size()  # 单机多卡：world_size代表有几块GPU
    if getattr(args, 'distributed', False):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                print(f"[rank={dist.get_rank()}] world_size={world_size} device={device}")
        except Exception:
            pass

    # if world_size > 8:  # 如果进程总数大于8，确保输出目录在HDFS上，并且HDFS路径的格式正确
    #     assert hexists(args.output_hdfs) and args.output_hdfs.startswith('hdfs'), "for collect_result among nodes"

    if args.bs > 0:  # 如果命令行参数中指定了bs（批处理大小），则更新配置中的训练批处理大小（按总batch除以卡数）
        config['batch_size_train'] = max(1, args.bs // max(world_size, 1))

    # 动态批大小已禁用（DDP 下不建议在线探测显存）

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt')) #日志地址
    logger.write(args.__repr__())

    # device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    # args.device = device

    #设置随机种子以确保实验的可重复性，并配置 CUDA 加速库以提升性能。同时设置当前 CUDA 设备为 args.gpu 指定的 GPU。
    args.seed = args.seed + xvlm_utils.get_rank()  # 设置随机种子，以确保实验可重复
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)   #XVLM
    random.seed(args.seed)  #XVLM
    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(args.gpu)

    #XVLM
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # 如果 args.dataset 是 'GQA'，则从文件 'data/gqa/dictionary.pkl' 中加载词典 dictionary，
    # 并初始化训练集 train_dset 和验证集 val_dset。
    if args.dataset == 'GQA':
        dictionary = Dictionary.load_from_file('data/gqa/dictionary.pkl')   #这个词典文件通常包含了数据集中出现的所有词汇，比如对象名称、属性描述等
        train_dset = GQAFeatureDataset(args, 'train', dictionary, adaptive=True,
                                       ann_file=config['train_file'], transform=transform,
                                       gqa_image_root=config['gqa_root'],
                                       gqa_answer_list=config['answer_list'], text_encoder=config['text_encoder'])  # XVLM
        val_dset = GQAFeatureDataset(args, 'val', dictionary, adaptive=True,
                                     ann_file=config['train_file'], transform=transform,
                                     gqa_image_root=config['gqa_root'],
                                     gqa_answer_list=config['answer_list'], text_encoder=config['text_encoder'])
    elif args.dataset == 'VQA':
        dictionary = Dictionary.load_from_file('data/vqa/dictionary.pkl')
        train_dset = VQAFeatureDataset(args,'train', dictionary, adaptive=True,
                                       ann_file=config['train_file'],transform=transform,vqa_image_root=config['vqa_root'],
                                       vg_image_root=config['vg_root'],
                                       answer_list=config['answer_list'],text_encoder=config['text_encoder'])   #XVLM
        val_dset = VQAFeatureDataset(args,'val', dictionary, adaptive=True,
                                     ann_file=config['train_file'], transform=transform,vqa_image_root=config['vqa_root'],
                                     vg_image_root= config['vg_root'],
                                     answer_list=config['answer_list'], text_encoder=config['text_encoder'])
    else:
        raise BaseException("Dataset name not found!")

    config['pad_token_id'] = train_dset.pad_token_id
    config['eos'] = train_dset.eos_token

    #根据 args.model 动态构建模型构造函数的名称 constructor，
    # 然后通过 getattr 函数从 base_model 模块中获取相应的模型构造函数，
    # 并使用 train_dset 和 args 初始化模型 model。
    # 将模型移动到指定的设备 device 上。
    batch_size = args.batch_size
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args,config)
    model = model.to(device)


    if args.distributed:  # 如果是分布式训练，包装模型以支持分布式数据并行
        # 关闭 broadcast_buffers 可减少 BN buffer 同步引发的不一致
        ddp_kwargs = dict(
            device_ids=[args.gpu],
            broadcast_buffers=False,
            bucket_cap_mb=getattr(args, 'ddp_bucket_cap_mb', 100)
        )
        # 严格 DDP：要求所有参数参与，提升性能
        if getattr(args, 'strict_ddp', False):
            ddp_kwargs['find_unused_parameters'] = False
        else:
            ddp_kwargs['find_unused_parameters'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    optim = None
    epoch = 0
    # load snapshot
    #如果指定了 args.input，则加载模型快照。 --input:（input file directory for continue training from stop one）
    if args.input is not None:
        if xvlm_utils.is_main_process():
            print(f'🔄 正在加载模型快照: {args.input}')
        
        # 使用 map_location 确保模型加载到正确的设备
        model_data = torch.load(args.input, map_location=device)
        
        # 加载模型状态字典
        if 'model_state' in model_data:
            model.load_state_dict(model_data['model_state'])
        else:
            model.load_state_dict(model_data)
        
        # 更新 epoch 信息
        epoch = model_data.get('epoch', 0) + 1
        
        # 清理加载的模型数据，释放内存
        del model_data
        memory_cleanup()
        
        if xvlm_utils.is_main_process():
            print(f'✅ 模型加载完成，将从 epoch {epoch} 开始继续训练')
            print(f'💾 当前显存占用: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB')

    #根据 args.use_both 的值，决定如何设置训练数据加载器 train_loader 和评估数据加载器 eval_loader。
    # 如果 args.use_both 为真，则将训练集和验证集合并为 trainval_dset，并创建相应的加载器。否则，分别创建训练集和验证集的加载器
    if args.use_both:  # use train & val splits to optimize
        trainval_dset = ConcatDataset([train_dset, val_dset])
        
        # 使用优化的批处理函数
        if xvlm_utils.is_main_process():
            print("🚀 使用优化的批处理函数 (compatible_collate_fn)")
        train_loader = DataLoader(
            trainval_dset, 
            batch_size, 
            shuffle=True, 
            num_workers=0, 
            collate_fn=compatible_collate_fn,
            pin_memory=True
        )
        eval_loader = None
        
        # 清理原始数据集，释放内存
        del train_dset, val_dset, trainval_dset
        memory_cleanup()
        
        if xvlm_utils.is_main_process():
            print(f'💾 合并数据集 DataLoader 创建完成，当前显存占用: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB')
    else:   #命令行中不加--use_both，默认为false
        datasets = [train_dset, val_dset]

        if args.distributed:  # 如果是分布式训练，获取分布式环境的进程数和全局排名
            num_tasks = xvlm_utils.get_world_size()  # 获取分布式环境中的进程总数
            global_rank = xvlm_utils.get_rank()  # 获取当前进程的排名
            samplers = create_sampler(datasets, [True, True], num_tasks, global_rank)  # 创建分布式采样器
        else:
            samplers = [None, None]  # 如果不是分布式训练，使用默认的None采样器

        train_dataset_size = len(train_dset)  # 训练和测试数据集的大小
        world_size = xvlm_utils.get_world_size()  # 获取进程总数

        if xvlm_utils.is_main_process():     # 如果是主进程，打印数据集大小和批处理大小信息
            logger.write(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")
            # print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")
        
        # 优化数据加载器配置（支持单 worker 调试）
        if getattr(args, 'dataloader_single_worker', False):
            num_workers = 0
            prefetch_factor = None
            persistent_workers = False
        else:
            num_workers = min(8, os.cpu_count())  # 根据CPU核心数调整worker数量
            prefetch_factor = 2  # 预取因子
            persistent_workers = True
        
        # 选择优化的批处理函数
        use_monitoring = getattr(args, 'enable_performance_monitoring', False)
        if use_monitoring:
            collate_fn = monitored_compatible_collate_fn
            if xvlm_utils.is_main_process():
                print("🚀 使用带性能监控的优化批处理函数")
        else:
            collate_fn = compatible_collate_fn
            if xvlm_utils.is_main_process():
                print("🚀 使用优化的批处理函数")
        
        # 构造训练 DataLoader 的参数，避免在单 worker 模式传入 prefetch_factor/persistent_workers
        train_dl_kwargs = dict(
            dataset=train_dset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=samplers[0],
            drop_last=getattr(args, 'train_drop_last', False),
        )
        if num_workers > 0:
            train_dl_kwargs.update(dict(
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
            ))
        else:
            train_dl_kwargs.update(dict(num_workers=0))
        train_loader = DataLoader(**train_dl_kwargs)

        # 构造评估 DataLoader 的参数
        eval_dl_kwargs = dict(
            dataset=val_dset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=samplers[1],
            drop_last=True,
        )
        if not getattr(args, 'dataloader_single_worker', False) and num_workers > 0:
            eval_dl_kwargs.update(dict(
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
            ))
        else:
            eval_dl_kwargs.update(dict(num_workers=0))
        eval_loader = DataLoader(**eval_dl_kwargs)

        # 验证各 rank 的 DataLoader 步数是否一致，避免不同步
        if getattr(args, 'distributed', False):
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    local_len = torch.tensor([len(train_loader)], device=device)
                    len_max = local_len.clone()
                    len_min = local_len.clone()
                    dist.all_reduce(len_max, op=dist.ReduceOp.MAX)
                    dist.all_reduce(len_min, op=dist.ReduceOp.MIN)
                    if xvlm_utils.is_main_process():
                        print(f"[dataloader] train_len(rank_local)={int(local_len.item())} max={int(len_max.item())} min={int(len_min.item())}")
            except Exception:
                pass
        
        if xvlm_utils.is_main_process():
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(f"🚀 优化的数据加载器配置:")
            print(f"  - Worker数量: {num_workers}")
            print(f"  - 预取因子: {prefetch_factor}")
            print(f"  - 持久化Worker: {persistent_workers}")
            print(f"  - 批处理函数: {'监控版优化函数' if use_monitoring else '标准优化函数'}")
            print(f"  - 训练集内存占用: {sys.getsizeof(train_dset)} bytes")
            print(f"  - 验证集内存占用: {sys.getsizeof(val_dset)} bytes")
            print("  ✓ 启用智能缓存机制")
            print("  ✓ 启用动态填充优化")
            print("  ✓ 启用多层错误处理")
            print("  ✓ 启用内存共享优化")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        # 清理临时变量，释放内存
        del train_dset, val_dset, datasets, samplers
        del train_dl_kwargs, eval_dl_kwargs
        memory_cleanup()
        
        if xvlm_utils.is_main_process():
            print(f'💾 DataLoader 创建完成，当前显存占用: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB')

    # 训练前最终内存清理和状态检查
    memory_cleanup()
    if xvlm_utils.is_main_process():
        print("=" * 60)
        print(f"🚀 开始训练 - 从 epoch {epoch} 开始")
        print(f"💾 训练前显存占用: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"💾 训练前显存缓存: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        print("=" * 60)
    
    train(args, model, train_loader, eval_loader, args.epochs, args.output, optim, epoch,config_xvlm=config)
    # eval_cfrf_score, fg_score, coarse_score, ens_score, mid_score, bound = evaluate(model, eval_loader, args,config)


