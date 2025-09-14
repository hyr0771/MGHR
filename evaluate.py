"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import sys
import json

import torch
from torch.utils.data import DataLoader

from src.FFOE.dataset import Dictionary, GQAFeatureDataset, VQAFeatureDataset
import src.FFOE.base_model as base_model
from src.FFOE.train import evaluate,predict
import src.utils as utils

import ruamel.yaml as yaml
from torchvision import transforms
import xvlm.utils as xvlm_utils
from xvlm.dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE CFRF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, default='saved_models/VQAv2_2',
                        help='input file directory for loading a model')
    parser.add_argument('--output', type=str, default='results/VQAv2_2',
                        help='output file directory for saving VQA answer prediction file')
    # Utilities
    parser.add_argument('--epoch', type=str, default='12',
                        help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')

    # Choices of models
    parser.add_argument('--model', type=str, default='CFRF_Model', choices=['CFRF_Model'],
                        help='the model we use')

    # INTERACTION LEARNING COMPONENTS HYPER-PARAMETERS------------------------------------------------------------------
    # BAN
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')
    parser.add_argument('--counter_act', type=str, default='zhang', choices=['zhang'],
                        help='the counter activation')

    # CONSTANT HYPER-PARAMETERS (Advanced hyper-params for testing, experimenting or fine-tuning)------------------------
    # Utilities - gpu
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    # Bounding box set
    parser.add_argument('--max_boxes', default=50, type=int, metavar='N',
                        help='number of maximum bounding boxes for K-adaptive')
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')

    # Stat word
    parser.add_argument('--num_stat_word', default=30, type=int, metavar='N',
                        help='number of statistical word')

    # Question embedding
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

    # Data
    parser.add_argument('--dataset', type=str, default='GQA', choices=['GQA', 'VQA'],
                        help='Dataset to train and evaluate')

    # Debugging
    parser.add_argument("--tiny", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    # 移除未使用的 LXMERT 相关参数

    # Optimization
    # 移除未使用的 mceLoss

    # Training configuration
    # 移除旧的 multiGPU 参数（使用 --distributed + torchrun）
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Fine-tuning arguments
    parser.add_argument('--omega_q', type=float, default=0.1,
                        help='omega for control the effect of question instructions')
    parser.add_argument('--omega_v', type=float, default=0.1,
                        help='omega for control the effect of image semantics')

    parser.add_argument('--topk', type=str, default='6')

    # Modal interaction hyperparameters
    parser.add_argument('--AOA_layers', type=int, default=6,
                        help='The number of layers of the AOA')

    # XVLM
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', default='./xvlm/configs/VQA_480.yaml')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--output_hdfs', type=str, default='', help="to collect eval results among nodes")

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--xvlm_seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--bs', default=-1, type=int)
    parser.add_argument('--evaluate', action='store_true')

    parser.add_argument("--general_hidden_dim", type=int, default=300)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--use_all_instrs", action="store_true", default=False)
    parser.add_argument("--use_global_mask", action="store_true", default=False)
    parser.add_argument("--node_classification", action="store_true", default=False)
    parser.add_argument("--sampler_type", type=str, default="simple")
    parser.add_argument("--sample_k", type=int, default=2)
    parser.add_argument("--nb_samples", default=1, type=int, metavar="N")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--beta", default=10.0, type=float)
    parser.add_argument("--tau", default=1.0, type=float)

    parser.add_argument("--gnn_gating", type=int, default=1)
    parser.add_argument("--use_instruction", type=int, default=1)
    parser.add_argument("--use_masking", type=int, default=1)
    parser.add_argument("--use_mgat", type=int, default=0)
    parser.add_argument("--mgat_masks", nargs="+", type=float, default=[1.0, 1.0, 1.0, 0.1])

    parser.add_argument("--use_topk", type=int, default=True)
    parser.add_argument("--interpretable_mode", type=int, default=False)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--concat_instr", type=int, default=0)
    parser.add_argument("--embed_cat", type=int, default=0)
    parser.add_argument("--text_sampling", action="store_true", default=False)
    parser.add_argument("--mgat_layers", default=4, type=int, metavar="N")
    parser.add_argument("--clip_url", default='/mnt/sda/hzh/.cache/huggingface/transformers/openai/clip-vit-base-patch32/',
                        help='url of clip-vit-base-patch32')

    # Return args
    args = parser.parse_args()
    return args

# 定义一个函数来处理无法直接序列化的对象
def convert(x):
    if hasattr(x, "tolist"):  # 检查对象是否有 tolist 方法，这是 PyTorch 张量的一个特性
        return x.tolist()  # 将张量转换为列表
    elif isinstance(x, dict):
        return {key: convert(value) for key, value in x.items()}  # 递归处理字典中的每个值
    elif isinstance(x, list):
        return [convert(element) for element in x]  # 递归处理列表中的每个元素
    else:
        return x  # 对于其他类型，直接返回

def save_results_to_json(results, filename):
    """通用函数：将结果保存为JSON文件"""
    print(f"开始保存{filename}文件....")
    with open(filename, 'w') as file:
        json.dump(results, file, ensure_ascii=False)
    print(f"保存{filename}文件完成！！")

if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()
    # XVLM
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    xvlm_utils.init_distributed_mode(args)  # 初始化分布式训练模式
    # 评测阶段：若为分布式，仅主进程执行，其他进程直接退出（避免重复写文件）
    if getattr(args, 'distributed', False) and not xvlm_utils.is_main_process():
        print('Non-main process exits evaluation.')
        sys.exit(0)
    device = torch.device(args.device)  # 设置设备，如果使用GPU训练，则为'cuda'
    world_size = xvlm_utils.get_world_size()  # 单机多卡：world_size代表有几块GPU

    # 设置cudnn加速
    torch.backends.cudnn.benchmark = True
    # args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    # torch.cuda.set_device(args.gpu)

    # XVLM
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    # 根据数据集类型加载相应的数据集和词典
    if args.dataset == 'GQA':
        dictionary = Dictionary.load_from_file('data/gqa/dictionary.pkl')

        test_dataset = GQAFeatureDataset(args, 'test', dictionary, adaptive=True,
                                       ann_file=config['test_file'], transform=test_transform,
                                       gqa_image_root=config['gqa_root'],
                                       gqa_answer_list=config['answer_list'],
                                       text_encoder=config['text_encoder'])  # XVLM
    elif args.dataset == 'VQA':
        dictionary = Dictionary.load_from_file('data/vqa/dictionary.pkl')
        test_dataset = VQAFeatureDataset(args, 'test2015', dictionary, adaptive=True,
                                       ann_file=config['test_file'], transform=test_transform,
                                       vqa_image_root=config['vqa_root'],
                                       vg_image_root=config['vg_root'],
                                       answer_list=config['answer_list'], text_encoder=config['text_encoder'])  # XVLM

    else:
        raise BaseException("Dataset name not found!")

    config['pad_token_id'] = test_dataset.pad_token_id
    config['eos'] = test_dataset.eos_token

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    if args.distributed:  # 如果是分布式训练，获取分布式环境的进程数和全局排名
        num_tasks = xvlm_utils.get_world_size()  # 获取分布式环境中的进程总数
        global_rank = xvlm_utils.get_rank()  # 获取当前进程的排名
        samplers = create_sampler([test_dataset], [False], num_tasks, global_rank)  # 创建分布式采样器
    else:
        samplers = [None]  # 如果不是分布式训练，使用默认的None采样器

    test_dataset_size = len(test_dataset)  # 训练和测试数据集的大小
    world_size = xvlm_utils.get_world_size()  # 获取进程总数

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, collate_fn=utils.trim_collate,
                              pin_memory=True,
                              sampler=samplers[0])

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(test_dataset, args, config)

    model_path = args.input + '/model_epoch%s.pth' % args.epoch
    print('loading %s' % model_path)
    model_data = torch.load(model_path,weights_only=True, map_location=args.device)


    # 获取 model_state
    model_state = model_data['model_state']
    # 去除键前的 'module.' 前缀
    new_model_state = {key.replace('module.', ''): value for key, value in model_state.items()}
    # model.load_state_dict(model_data.get('model_state', model_data))
    model.load_state_dict(new_model_state)
    model = model.to(device)

    if args.split == 'val':     #TODO 可能需要修改
        print("Evaluating val ...")
        model.train(False)
        eval_cfrf_score, eval_fg_score, eval_coarse_score, eval_ens_score, bound = evaluate(model, test_loader, args, config_xvlm=config)
        print('\tCFRF score: %.2f (%.2f)' % (100 * eval_cfrf_score, 100 * bound))
        print('\tFG score: %.2f (%.2f)' % (100 * eval_fg_score, 100 * bound))
        print('\tCoarse score: %.2f (%.2f)' % (100 * eval_coarse_score, 100 * bound))
        print('\tEns score: %.2f (%.2f)' % (100 * eval_ens_score, 100 * bound))

    elif args.split == 'test2015':  #VQA
        print("predict test ...")
        model.train(False)
        # 多卡时仅主进程执行预测与文件保存
        if not xvlm_utils.is_main_process():
            return

        # cfrf_quesid2ans, fg_quesid2ans, ens_quesid2ans ,cg_quesid2ans = predict(model,test_loader, args, config_xvlm=config)
        cfrf_quesid2ans, fg_quesid2ans, ens_quesid2ans, cg_quesid2ans, mid_quesid2ans = predict(model, test_loader,
                                                                                                args,
                                                                                                config_xvlm=config)

        results = []
        for quesid in cfrf_quesid2ans:
            result = {}
            result['question_id'] = quesid
            result['answer'] = cfrf_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存cfrf_quesid2ans.json文件....")
            with open("./cfrf_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存cfrf_quesid2ans.json文件完成！！")

        results.clear()
        for quesid in fg_quesid2ans:
            result = {}
            result['question_id'] = quesid
            result['answer'] = fg_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存fg_quesid2ans.json文件....")
            with open("./fg_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存fg_quesid2ans.json文件完成！！")

        results.clear()
        for quesid in cg_quesid2ans:
            result = {}
            result['question_id'] = quesid
            result['answer'] = cg_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存cg_quesid2ans.json文件....")
            with open("./cg_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存cg_quesid2ans.json文件完成！！")

        results.clear()
        for quesid in ens_quesid2ans:
            result = {}
            result['question_id'] = quesid
            result['answer'] = ens_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存ens_quesid2ans.json文件....")
            with open("./ens_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存ens_quesid2ans.json文件完成！！")

        results.clear()
        for quesid in mid_quesid2ans:
            result = {}
            result['question_id'] = quesid
            result['answer'] = mid_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存mid_quesid2ans.json文件....")
            with open("./mid_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存mid_quesid2ans.json文件完成！！")
    
    elif args.split == 'test': #GQA数据集
        print("predict test ...")
        model.train(False)
        if not xvlm_utils.is_main_process():
            return
        cfrf_quesid2ans, fg_quesid2ans, ens_quesid2ans, cg_quesid2ans, mid_quesid2ans = predict(model,test_loader, args, config_xvlm=config)

        results = []
        for quesid in cfrf_quesid2ans:
            result = {}
            result['questionId'] = quesid
            result['prediction'] = cfrf_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存cfrf_quesid2ans.json文件....")
            with open("./cfrf_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存cfrf_quesid2ans.json文件完成！！")

        results.clear()
        for quesid in fg_quesid2ans:
            result = {}
            result['questionId'] = quesid
            result['prediction'] = fg_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存fg_quesid2ans.json文件....")
            with open("./fg_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存fg_quesid2ans.json文件完成！！")

        results.clear()
        for quesid in ens_quesid2ans:
            result = {}
            result['questionId'] = quesid
            result['prediction'] = ens_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存ens_quesid2ans.json文件....")
            with open("./ens_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存ens_quesid2ans.json文件完成！！")

        results.clear()
        for quesid in cg_quesid2ans:
            result = {}
            result['questionId'] = quesid
            result['prediction'] = cg_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存cg_quesid2ans.json文件....")
            with open("./cg_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存cg_quesid2ans.json文件完成！！")

        results.clear()
        for quesid in mid_quesid2ans:
            result = {}
            result['questionId'] = quesid
            result['prediction'] = mid_quesid2ans[quesid]
            results.append(result)
        if xvlm_utils.is_main_process():
            print("开始保存mid_quesid2ans.json文件....")
            with open("./mid_quesid2ans.json", 'w') as file:
                json.dump(results, file, ensure_ascii=False)
            print("保存mid_quesid2ans.json文件完成！！")

