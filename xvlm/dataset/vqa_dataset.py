import os
import json
import random
from random import random as rand

from PIL import Image
from torch.utils.data import Dataset
# from dataset.utils import pre_question
from xvlm.dataset.utils import pre_question
from torchvision.transforms.functional import hflip

from transformers import BertTokenizer, RobertaTokenizer

class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, split="train", max_ques_words=30, answer_list='',
                 text_encoder='', use_roberta=False):

        self.careful_hflip = True

        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))  #[ vqa_train.json, vqa_val.json, vg_qa.json ] or [vqa_test.json]   annotation

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words

        tokenizer = RobertaTokenizer.from_pretrained(text_encoder) if use_roberta else \
            BertTokenizer.from_pretrained(text_encoder)

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = '</s>' if use_roberta else '[SEP]'

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, 'r'))
        
    def __len__(self):
        return len(self.ann)

    def left_or_right_in(self, question, answer):
        def _func(s):
            if ('left' in s) or ('right' in s):
                return True
            else:
                return False

        if _func(question):
            return True

        if isinstance(answer, list):
            for ans in answer:
                if _func(ans):
                    return True
        else:
            if _func(answer):
                return True

        return False

    def __getitem__(self, index):

        ann = self.ann[index]

        if 'dataset' in ann.keys(): # 根据注释信息中的dataset字段确定图像的存储位置，并构造图像的完整【路径】
            if ann['dataset'] == 'vqa':
                image_path = os.path.join(self.vqa_root, ann['image'])
            elif ann['dataset'] == 'vg':
                image_path = os.path.join(self.vg_root, ann['image'])
            elif ann['dataset'] == 'gqa':
                image_path = ann['image']
            else:
                raise NotImplementedError

        else:
            image_path = os.path.join(self.vqa_root, ann['image'])

        image = Image.open(image_path).convert('RGB')   # 打开图像文件，并将其转换为RGB格式

        # 如果当前不是测试集（test），并且随机数小于0.5，则对图像进行水平翻转
        # 如果设置了仔细的水平翻转（careful_hflip），并且问题或答案中含有'left'或'right'，则不进行翻转
        if (self.split != 'test') and rand() < 0.5:
            if self.careful_hflip and self.left_or_right_in(ann['question'], ann['answer']):
                pass
            else:
                image = hflip(image)

        # 对图像进行预处理（例如缩放、归一化等），使用初始化时传入的transform
        image = self.transform(image)

        # 如果当前是测试集（test），则只返回图像、预处理后的问题和问题ID
        if self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['question_id']
            return image, question, question_id

        # 如果当前是训练集（train），则返回图像、预处理后的问题、可能的答案列表以及对应的权重
        elif self.split == 'train':
            question = pre_question(ann['question'], self.max_ques_words)

            if ('dataset' in ann.keys()) and (ann['dataset'] == 'vg'):  # 对于来自VG数据集的样本，只包含一个答案，并赋予相等的权重
                answers = [ann['answer']]
                weights = [0.5]

            else:       # 对于来自VQA数据集的样本，可能包含多个答案，计算每个答案的权重
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1 / len(ann['answer'])
                    else:
                        answer_weight[answer] = 1 / len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())
            # 将每个答案后面添加EOS标记
            answers = [answer + self.eos_token for answer in answers]  # fix bug

            return image, question, answers, weights

        else:
            raise NotImplementedError
