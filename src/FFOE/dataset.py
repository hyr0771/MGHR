"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import src.utils as utils
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import torch
from torch.utils.data import Dataset
import itertools

from transformers import BertTokenizer, RobertaTokenizer
from PIL import Image
from xvlm.dataset.utils import pre_question

from ISubGVQA.datasets.scene_graph import GQASceneGraphs,VQASceneGraphs
from transformers import CLIPTokenizerFast, CLIPTokenizer

COUNTING_ONLY = False

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering


#可能用于问题类型或答案的筛选。
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

'''
用于创建和管理词汇表，包括单词到索引和索引到单词的映射，以及文本分词功能。
'''
class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

"""
============================
load GQA Dataset
============================
"""
#定义_create_entry函数，用于创建数据集条目。
def _create_entry(img, question, answer, entity, teacher_logit):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer,
        'entity'      : entity,     ##多的两个
        'teacher_logit': teacher_logit}     ##多的两个
    return entry


#定义_load_gqa_dataset函数，用于加载GQA数据集。
"""
    Load entries from the GQA dataset.

    参数:
    - dataroot: 数据集的根目录路径。
    - args: 包含加载数据集所需的配置参数的对象。
    - name: 数据集的名称，可以是 'train', 'val', 'test-dev2015', 'test2015' 等。
    - img_id2val: 一个字典，将图像ID映射到它们的值，这些值可以用于检索图像或特征。
    (img_id2val: dict {img_id -> val} val can be used to retrieve image or features)

    返回:
    - entries: 包含数据集条目的列表。
"""
def _load_gqa_dataset(dataroot, args, name, img_id2val):
    """
    gqa_train_questions_entities.json:
        {"questions": [
        
        {"image_id": "2354786", 
        "question": "Is the sky dark?",
        "question_id": "02930152",
        "entities": ["dark", "sky"]}, 
    
        {"image_id": "2368326", 
         "question": "Is the tall clock small or large?", 
         "question_id": "15736264", 
         "entities": ["large", "small", "clock", "tall"]},
                    ....]}
        另外：entities:实体   entries:条目
    """
    question_path = os.path.join(
        dataroot, 'gqa_%s_questions_entities.json' % name)
    # 加载JSON格式的问题数据。
    #json.load(open(question_path)) 将JSON格式的数据转换成了Python字典
    # ..['questions'] 拿到字典{"questions":[...]} 给定的列表
    # sorted(...): 这个内置函数对提取出来的问题列表进行排序。
    #key=lambda x: x['question_id'] 是指按照 "question_id"对应的值的大小排序
    #最后得到按"question_id"值的大小排序后的列表
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    if 'test' != name[:4]:  # train, val
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)    # 构造答案数据的文件路径。
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])   #同样按照"question_id"对应的值的大小排序，和questions对应
        utils.assert_eq(len(questions), len(answers))
        entries = []

        # Train and evaluate on tiny sample
        if args.tiny:       # 如果配置参数args.tiny为True，则只加载小样本。
            questions = questions[:30000]
            answers = answers[:30000]

        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            entity = question['entities']

            entries.append(_create_entry(img_id2val[img_id], question, answer, entity, None))
    else:  # test
        entries = []
        for question in questions:
            img_id = question['image_id']
            entity = question['entities']
            entries.append(_create_entry(img_id2val[img_id], question, None, entity, None))

    return entries

class GQAFeatureDataset(Dataset):
    def __init__(self, args, name, dictionary,ann_file, transform, gqa_image_root, gqa_answer_list, text_encoder, dataroot='data/gqa', adaptive=False):  #adaptive: 是否自适应 main函数中：adaptive=true
        super(GQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015', 'test'] #使用断言确保 name 参数在预定义的数据集名称列表中。

        #加载答案到标签的映射和标签到答案的映射，分别存储在 ans2label 和 label2ans 中。
        #num_ans_candidates 记录了答案候选的数量，即 ans2label 的长度。
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.max_boxes = args.max_boxes #设置最大图像框数和问题的最大长度。  '--max_boxes', default=50
        self.question_len = args.question_len

        self.dictionary = dictionary
        self.adaptive = adaptive
        # self.teacher_logits = [] #这个没有用上
        print('Create %s entries' % name)  #打印输出当前数据集的名称

        # load stat_word
        """
         【stat_words】相关的 JSON 文件 和 【stat_skip_imgid】
        stat_words内容: （Statistical Words）--> 统计词汇
        stat_words内容: 
        "2375363": "plate,white,tablecloth,meat,........knife,wine glasses,pan,glass,wine glass",
            "2323421": "glasses,boys", 
            "713255": "", 
        stat_skip_imgid:
            ["2325191", "2325332", "2329767", "2339279", ... ]
        【现象】
            train_6_stat_words.json 里面有的键对应的值为空。
            train_6_stat_skip_imgid.json 存的都是值为空的id，但是不全，有的在stat_words值为空，但是skip_imgid 里面没有。
        
            train_6_stat_words.json 里面的词很多都来自于 train_predicates.json，可能就是论文里面所提到的“谓词”(predicate)
            另外，train_6_stat_skip_imgid.json 里面的ID在 train_predicates.json也能找到，内容为：
                "2330612": [["location, is, outdoors"]]
                "2399245": [["global, is, background"]] 
        """
        # 兼容缺失文件的情况，提供空回退，避免训练中断
        try:
            self.stat_words = json.load(open('data/gqa/%s_%s_stats_words.json' % (name, args.topk))) # args.topk default='6'
        except Exception:
            self.stat_words = {}
        try:
            self.stat_skip_imgid = json.load(open('data/gqa/%s_%s_stats_skip_imgid.json' % (name, args.topk)))
        except Exception:
            self.stat_skip_imgid = []
        self.stat_features = {}

        # load attribute word
        """
        【属性词汇】相关的 JSON 文件和【跳过图像】 ID 的信息
        {"2370799": ["blue helmet", "blue bike", "tall grass", "riding man"], "2370791": ["white bowl", "red box", "silver spoon", "silver faucet", "brown container", "brown faucet"],... }
        ["2325191", "2323928",..] (与上面的stat_skip_imgid 内容一样，但是顺序不一样)
        
        predivates.json -> "2370799": [["global, is, background"], ["attribute, bag, black", "to the left of, bag, men", "to the left of, bag, man"], ["to the right of, helmet, men"], ["attribute, grass, tall"], ["attribute, bike, blue", "to the left of, bike, bike", "to the left of, bike, man"], ["attribute, bike, orange", "to the right of, bike, bike", "to the right of, bike, men"], ["attribute, helmet, blue"], ["to the right of, man, men", "wearing, man, helmet", "to the right of, man, bike", "riding, man, bike", "to the right of, man, bag"], ["to the left of, men, bike", "to the right of, men, bag", "to the left of, men, helmet", "to the left of, men, man", "riding, men, bike"]]
        貌似训练没有用到这个attr
        
        """
        try:
            self.attr_words = json.load(open('data/gqa/%s_attr_words_non_plural_words.json' % name))
        except Exception:
            self.attr_words = {}
        try:
            self.attr_skip_imgid = json.load(open('data/gqa/%s_attr_skip_imgid.json' % name))
        except Exception:
            self.attr_skip_imgid = []
        self.skip_imgid = []
        self.attr_features = {}

        self.ans_list = []

        # 加载图像 ID 到索引的映射，存储在 img_id2idx 中
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))

        # Load image feature
        #使用 HDF5 文件加载图像特征、空间特征和位置框信息，
        # 并存储在 features、spatials 和 pos_boxes 中。
        h5_path = os.path.join(dataroot, '%s.hdf5' % name)
        print('loading features from h5 file %s ' % h5_path)
        with h5py.File(h5_path, 'r') as hf:
            # image_bb.shape (2222393, 4)
            self.features = np.array(hf.get('image_features'))  # (2222393, 2048)
            self.spatials = np.array(hf.get('spatial_features')) #(2222393, 6)
            self.pos_boxes = np.array(hf.get('pos_boxes'))  #(72140, 2)

        #调用多个方法进行数据预处理
        self.entries = _load_gqa_dataset(dataroot, args, name, self.img_id2idx) # 函数解析在上面  返回GQA数据集的条目（entries）
        self.tokenize(self.question_len)    #tokenize：对问题进行标记化。
        self.ISubGVQA_tokenizer = CLIPTokenizerFast.from_pretrained(args.clip_url, use_fast=True)

        self.stat_word_tokenize_1(args.num_stat_word) #stat_word_tokenize_1：对统计词汇进行标记化。  args.num_stat_word -> default=30
        self.attr_word_tokenize(15) #attr_word_tokenize：对属性词汇进行标记化。
        self.ans_tokenize() #ans_tokenize：对答案进行标记化。
        self.entity_tokenize() #entity_tokenize：对实体进行标记化。
        self.tensorize() #tensorize：将数据转换为张量表示。
        self.v_dim = self.features.size(1) #v_dim 存储 features 的特征维度。
        self.s_dim = self.spatials.size(1) #s_dim 存储 spatials 的特征维度。

        # XVLM

        self.ann = []
        for f in ann_file:
            self.ann += json.load(
                open(f, 'r'))  # [ vqa_train.json, vqa_val.json, vg_qa.json ] or [vqa_test.json]   annotation
        self.ann_dict = {item["question_id"]: item for item in
                         self.ann}  # 使用字典推导式 {item["question_id"]: item for item in data} 将 question_id 作为键，整个对象作为值,查找对象时只需要O(1)
        del self.ann


        self.transform = transform
        self.gqa_root = gqa_image_root
        # 为兼容 XVLM 注释中的 dataset 字段（vqa/vg），提供回退根目录
        self.vqa_root = None
        self.vg_root = None

        self.tokenizer = BertTokenizer.from_pretrained(text_encoder)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token = '[SEP]'

        self.max_ques_words = 50
        self.answer_list = json.load(open(gqa_answer_list, 'r'))

        #ISubGVQA（GQA dataset）
        self.sg_feature_lookup = GQASceneGraphs()  # ISubGVQA
        # self.sg_cache = dict()
        
        # 在 __getitem__ 中需要用到的长度常量
        self.num_stat_word = args.num_stat_word


    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        将问题文本转换为标记序列。
         这将在数据集中的每个条目添加一个q_token字段。
         -1代表空值，在嵌入时应该被视为填充索引
        """
        for entry in self.entries:
            # 使用dictionary的tokenize方法对问题进行【分词】，False参数表示不进行特殊处理
            #entry['question'] --> "Is the sky dark?"
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]    #截断标记列表，使其长度不超过max_length
            if len(tokens) < max_length:
                # 如果标记列表长度小于max_length，则在句子前面进行填充
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding   #创建一个填充列表，填充值为dictionary定义的padding_idx
            utils.assert_eq(len(tokens), max_length)    # 确保标记列表的长度等于max_length，如果不相等则抛出异常
            entry['q_token'] = tokens  #将最终的标记列表赋值给 entry['q_token']，这样每个条目就包含了一个标准化长度的问题标记列表。

    def entity_tokenize(self, max_length=7):
        """Tokenizes the instruction word.

        This will add entity_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding

        该方法将在数据集的每个条目中添加一个 'entity_token' 列表。
        'entity_token' 列表包含了实体文本的分词形式。
        如果实体的标记数少于 'max_length'，则列表会用填充索引填充，
        以确保所有实体在批处理中具有相同的长度。
        """
        for entry in self.entries:
            #entry['entity']  -内容->  "entities": ["large", "small", "clock", "tall"]
            entity = entry['entity']
            entity = ' '.join(entity) # 将实体列表转换为一个由空格分隔的字符串。
            tokens = self.dictionary.tokenize(entity, False) # 使用dictionary的tokenize方法对实体字符串进行分词。
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding

            entry['entity_token'] = tokens

    def ans_tokenize(self, max_length=2):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            try:
                # 尝试从条目的答案中获取标签，并将其作为答案。
                ans = self.label2ans[entry['answer']['labels'][0]]
                tokens = self.dictionary.tokenize(ans, False)  # 使用dictionary的tokenize方法对答案字符串进行分词。
            except:
                tokens = []

            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['ans_token'] = tokens

    # Tokenize statistical words 2-gram
    def stat_word_tokenize(self, max_length=40):
        for img_id in self.stat_words:
            words = self.stat_words[img_id]
            # words = words.split(',')
            words = words[:max_length]
            token_words = []
            for word in words:
                tokens = self.dictionary.tokenize(word, False)
                tokens = tokens[:2]
                if len(tokens) < 2:
                    padding = [self.dictionary.padding_idx] * (2 - len(tokens))
                    tokens = tokens + padding
                token_words.append(tokens)
            if len(words) < max_length:
                tmp = list(np.full(2, self.dictionary.padding_idx))
                tmp_token_words = [tmp for _ in range(max_length - len(words))]
                token_words += tmp_token_words
            self.stat_features[img_id] = token_words

    # Tokenize attribute words
    def attr_word_tokenize(self, max_length=15):
        for img_id in self.attr_words:
            words = self.attr_words[img_id]
            words = words[:max_length]
            token_words = []
            for word in words:
                tokens = self.dictionary.tokenize(word, False)
                tokens = tokens[:3]
                if len(tokens) < 3:
                    padding = [self.dictionary.padding_idx] * (3 - len(tokens))
                    tokens = tokens + padding
                token_words.append(tokens)
            if len(words) < max_length:
                tmp = list(np.full(3, self.dictionary.padding_idx))
                tmp_token_words = [tmp for _ in range(max_length - len(words))]
                token_words += tmp_token_words
            self.attr_features[img_id] = token_words

    # Tokenize statistical words
    def stat_word_tokenize_1(self, max_length=40):
        for img_id in self.stat_words:
            words = self.stat_words[img_id]
            words = words.split(',')
            words = ' '.join(words)
            tokens = self.dictionary.tokenize(words, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            self.stat_features[img_id] = tokens

    def ans_word_tokenize(self, max_length=2):
        ans_list = []
        for ans in self.label2ans:
            tokens = self.dictionary.tokenize(ans, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            ans_list.append(tokens)
        self.ans_list = ans_list

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            entity = torch.from_numpy(np.array(entry['entity_token']))
            entry['entity_token'] = entity
            ans = torch.from_numpy(np.array(entry['ans_token']))
            entry['ans_token'] = ans

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        """
        entry = {
            'question_id' : question['question_id'],
            'image_id'    : question['image_id'],
            'image'       : img,
            'question'    : question['question'],
            'answer'      : answer,
            'entity'      : entity,     ##多的两个
            'teacher_logit': teacher_logit}     ##多的两个

        features, spatials, stat_features, entity, attr_features, question, sent, target, ans
        v,           b,         w,           e,         attr,       q,        s,     a,    ans
        """
        entry = self.entries[index]
        features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        features = features[:self.max_boxes]  #self.max_boxes, default=50
        spatials = spatials[:self.max_boxes]

        question_ban = entry['q_token']
        sent = entry['question']
        entity = entry['entity_token']
        question_id = entry['question_id']
        answer = entry['answer']
        img_id = str(entry['image_id'])
        # 兼容缺失统计/属性特征，使用 padding 构造回退
        if img_id in self.stat_features:
            stat_feat_np = np.array(self.stat_features[img_id])
        else:
            # 回退为长度 num_stat_word、值为 padding_idx 的序列
            stat_feat_np = np.array([self.dictionary.padding_idx] * self.num_stat_word)
        stat_features = torch.from_numpy(stat_feat_np)
        if img_id in self.attr_features:
            attr_feat_np = np.array(self.attr_features[img_id])
        else:
            # 属性特征当前训练未使用，给定最小回退占位
            attr_feat_np = np.zeros((0,), dtype=np.int64)
        attr_features = torch.from_numpy(attr_feat_np)
        ans = entry['ans_token']

        # XVLM
        # 根据question_id来查找对应的XVLM数据
        ann = self.ann_dict.get(question_id)
        if 'dataset' in ann.keys():  # 根据注释信息中的dataset字段确定图像的存储位置，并构造图像的完整【路径】
            ds = ann['dataset']
            if ds == 'vqa' and self.vqa_root is not None:
                image_path = os.path.join(self.vqa_root, ann['image'])
            elif ds == 'vg' and self.vg_root is not None:
                image_path = os.path.join(self.vg_root, ann['image'])
            else:
                # 回退到 gqa_root，避免由于 root 缺失导致的路径错误
                image_path = os.path.join(self.gqa_root, ann['image'])
        else:
            image_path = os.path.join(self.gqa_root, ann['image'])

        image = Image.open(image_path).convert('RGB')  # 打开图像文件，并将其转换为RGB格式
        # 对图像进行预处理（例如缩放、归一化等），使用初始化时传入的transform
        image = self.transform(image)
        question_xvlm = pre_question(ann['question'], self.max_ques_words)

        sg_datum = self.sg_feature_lookup.query_and_translate(img_id)
        sg_datum.x = sg_datum.x.squeeze()
        sg_datum.edge_attr = sg_datum.edge_attr.squeeze()
        scene_graph = sg_datum
        # self.sg_cache[img_id] = scene_graph

        # question_id_xvlm = ann['question_id']
        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            # v, b, w, e, q, a, ans, image, question, question_id, scene_graph
            return features, spatials, stat_features, entity, question_ban, target, ans, image, question_xvlm, question_id,scene_graph
            # return target,image, question_xvlm,scene_graph
        else:  # test
            return features, spatials, stat_features, entity, question_ban, image, question_xvlm,question_id,scene_graph
            # return image, question_xvlm,scene_graph

    def __len__(self):
        return len(self.entries)

"""
============================
load VQA v2 Dataset
============================
"""
# def _create_entry_for_VQA(img, question, answer):
#     if None!=answer:
#         answer.pop('image_id')
#         answer.pop('question_id')
#     entry = {
#         'question_id' : question['question_id'],
#         'image_id'    : question['image_id'],
#         'image'       : img,
#         'question'    : question['question'],
#         'answer'      : answer}
#     return entry
def _load_vqa_dataset(dataroot,args, name, img_id2val):
    # print(img_id2val)
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """

    """
    v2_OpenEnded_mscoco_train2014_questions.json:
        "info": {...},
        "data_subtype": "train2014",
        "questions": [
            {
                "image_id": 458752,
                "question": "What is this photo taken looking through?",
                "question_id": 458752000
            },
    VQAv2_train_questions_entities.json
    [
        {
            "image_id": 9,
            "question": "How many cookies can be seen?",
            "question_id": 9000,
            "entities": [
                "cookies"
            ]
        }, ..
    ]
    """
    question_path = os.path.join(
        dataroot, 'VQAv2_%s_questions_entities.json' % name)
    print("==> question_path: {}".format(question_path))
    questions = sorted(json.load(open(question_path)),key=lambda x: x['question_id'])
    if 'test'!=name[:4]: # train, val
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])
        utils.assert_eq(len(questions), len(answers))
        entries = []

        # Train and evaluate on tiny sample
        # if args.tiny:  # 如果配置参数args.tiny为True，则只加载小样本。
        #     questions = questions[:30000]
        #     answers = answers[:30000]

        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            entity = question['entities']

            entries.append(_create_entry(img_id2val[img_id], question, answer, entity, None))
    else: # test2015
        entries = []
        for question in questions:
            img_id = question['image_id']
            entity = question['entities']
            entries.append(_create_entry(img_id2val[img_id], question, None, entity, None))

    return entries

class VQAFeatureDataset(Dataset):
    def __init__(self,args,name, dictionary,ann_file,transform,vqa_image_root,vg_image_root,answer_list,text_encoder, dataroot='data/vqa', adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.max_boxes = args.max_boxes  # 设置最大图像框数和问题的最大长度。  '--max_boxes', default=50
        self.question_len = args.question_len

        self.dictionary = dictionary
        self.adaptive = adaptive
        # self.teacher_logits = []  # 这个没有用上
        print('Create %s entries' % name)  # 打印输出当前数据集的名称
        # load stat_word
        self.stat_words = json.load(
            open('data/vqa/%s_VQAv2_stats_words.json' % (name)))
        # self.stat_skip_imgid = json.load(open('data/gqa/%s_%s_stats_skip_imgid.json' % (name, args.topk)))
        self.stat_features = {}
        # load attribute word
        #...
        self.attr_features = {}
        self.ans_list = []

        # 加载图像 ID 到索引的映射，存储在 img_id2idx 中
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))

        h5_path = os.path.join(dataroot, '%s%s.hdf5' % (name, '' if self.adaptive else '36'))
        print('loading features from h5 file')
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            self.pos_boxes = np.array(hf.get('pos_boxes'))
        """
        #调用多个方法进行数据预处理
        self.entries = _load_gqa_dataset(dataroot, args, name, self.img_id2idx) # 函数解析在上面  返回GQA数据集的条目（entries）
        self.tokenize(self.question_len)    #tokenize：对问题进行标记化。
        self.stat_word_tokenize_1(args.num_stat_word) #stat_word_tokenize_1：对统计词汇进行标记化。  args.num_stat_word -> default=30
        
        self.attr_word_tokenize(15) #attr_word_tokenize：对属性词汇进行标记化。
        self.ans_tokenize() #ans_tokenize：对答案进行标记化。
        self.entity_tokenize() #entity_tokenize：对实体进行标记化。
        self.tensorize() #tensorize：将数据转换为张量表示。
        self.v_dim = self.features.size(1) #v_dim 存储 features 的特征维度。
        self.s_dim = self.spatials.size(1) #s_dim 存储 spatials 的特征维度。
        """
        self.entries = _load_vqa_dataset(dataroot,args,name,self.img_id2idx)
        self.tokenize(self.question_len)
        self.stat_word_tokenize_1(args.num_stat_word)  #  args.num_stat_word -> default=30
        # self.attr_word_tokenize(15)  # attr_word_tokenize：对属性词汇进行标记化。
        self.ans_tokenize()  # ans_tokenize：对答案进行标记化。
        self.entity_tokenize()  # entity_tokenize：对实体进行标记化。
        self.tensorize()
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.spatials.size(1 if self.adaptive else 2)

        #XVLM

        self.ann = []
        for f in ann_file:
            self.ann += json.load(
                open(f, 'r'))  # [ vqa_train.json, vqa_val.json, vg_qa.json ] or [vqa_test.json]   annotation
        self.ann_dict = {item["question_id"]: item for item in self.ann}    #使用字典推导式 {item["question_id"]: item for item in data} 将 question_id 作为键，整个对象作为值,查找对象时只需要O(1)

        self.transform = transform
        self.vqa_root = vqa_image_root
        self.vg_root = vg_image_root

        self.tokenizer = BertTokenizer.from_pretrained(text_encoder)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token = '[SEP]'

        self.max_ques_words = 50
        self.answer_list = json.load(open(answer_list, 'r'))

        # ISubGVQA（VQAv2 dataset）
        self.sg_feature_lookup = VQASceneGraphs()  # ISubGVQA



    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        将问题文本转换为标记序列。
         这将在数据集中的每个条目添加一个q_token字段。
         -1代表空值，在嵌入时应该被视为填充索引
        """
        for entry in self.entries:
            # 使用dictionary的tokenize方法对问题进行【分词】，False参数表示不进行特殊处理
            #entry['question'] --> "Is the sky dark?"
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]    #截断标记列表，使其长度不超过max_length
            if len(tokens) < max_length:
                # 如果标记列表长度小于max_length，则在句子前面进行填充
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding   #创建一个填充列表，填充值为dictionary定义的padding_idx
            utils.assert_eq(len(tokens), max_length)    # 确保标记列表的长度等于max_length，如果不相等则抛出异常
            entry['q_token'] = tokens  #将最终的标记列表赋值给 entry['q_token']，这样每个条目就包含了一个标准化长度的问题标记列表。

    def entity_tokenize(self, max_length=7):
        """Tokenizes the instruction word.

        This will add entity_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding

        该方法将在数据集的每个条目中添加一个 'entity_token' 列表。
        'entity_token' 列表包含了实体文本的分词形式。
        如果实体的标记数少于 'max_length'，则列表会用填充索引填充，
        以确保所有实体在批处理中具有相同的长度。
        """
        for entry in self.entries:
            #entry['entity']  -内容->  "entities": ["large", "small", "clock", "tall"]
            entity = entry['entity']
            entity = ' '.join(entity) # 将实体列表转换为一个由空格分隔的字符串。
            tokens = self.dictionary.tokenize(entity, False) # 使用dictionary的tokenize方法对实体字符串进行分词。
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding

            entry['entity_token'] = tokens

    def ans_tokenize(self, max_length=2):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            try:
                # 尝试从条目的答案中获取标签，并将其作为答案。
                ans = self.label2ans[entry['answer']['labels'][0]]
                tokens = self.dictionary.tokenize(ans, False)  # 使用dictionary的tokenize方法对答案字符串进行分词。
            except:
                tokens = []

            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['ans_token'] = tokens

    # Tokenize statistical words 2-gram
    def stat_word_tokenize(self, max_length=40):
        for img_id in self.stat_words:
            words = self.stat_words[img_id]
            # words = words.split(',')
            words = words[:max_length]
            token_words = []
            for word in words:
                tokens = self.dictionary.tokenize(word, False)
                tokens = tokens[:2]
                if len(tokens) < 2:
                    padding = [self.dictionary.padding_idx] * (2 - len(tokens))
                    tokens = tokens + padding
                token_words.append(tokens)
            if len(words) < max_length:
                tmp = list(np.full(2, self.dictionary.padding_idx))
                tmp_token_words = [tmp for _ in range(max_length - len(words))]
                token_words += tmp_token_words
            self.stat_features[img_id] = token_words

    # Tokenize attribute words
    def attr_word_tokenize(self, max_length=15):
        for img_id in self.attr_words:
            words = self.attr_words[img_id]
            words = words[:max_length]
            token_words = []
            for word in words:
                tokens = self.dictionary.tokenize(word, False)
                tokens = tokens[:3]
                if len(tokens) < 3:
                    padding = [self.dictionary.padding_idx] * (3 - len(tokens))
                    tokens = tokens + padding
                token_words.append(tokens)
            if len(words) < max_length:
                tmp = list(np.full(3, self.dictionary.padding_idx))
                tmp_token_words = [tmp for _ in range(max_length - len(words))]
                token_words += tmp_token_words
            self.attr_features[img_id] = token_words

    # Tokenize statistical words
    def stat_word_tokenize_1(self, max_length=40):
        for img_id in self.stat_words:
            words = self.stat_words[img_id]
            words = words.split(',')
            words = ' '.join(words)
            tokens = self.dictionary.tokenize(words, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            self.stat_features[img_id] = tokens

    def ans_word_tokenize(self, max_length=2):
        ans_list = []
        for ans in self.label2ans:
            tokens = self.dictionary.tokenize(ans, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            ans_list.append(tokens)
        self.ans_list = ans_list


    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            entity = torch.from_numpy(np.array(entry['entity_token']))
            entry['entity_token'] = entity
            ans = torch.from_numpy(np.array(entry['ans_token']))
            entry['ans_token'] = ans

            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        """
        VQA!!
        entry = {
            'question_id' : question['question_id'],
            'image_id'    : question['image_id'],
            'image'       : img,
            'question'    : question['question'],
            'answer'      : answer,
            'entity'      : entity,     ##多的两个
            'teacher_logit': teacher_logit}     ##多的两个

        features, spatials, stat_features, entity, attr_features, question, sent, target, ans
        v,           b,         w,           e,         attr,       q,        s,     a,    ans
        """
        entry = self.entries[index]
        features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        features = features[:self.max_boxes]  # self.max_boxes, default=50
        spatials = spatials[:self.max_boxes]

        question_ban = entry['q_token']
        sent = entry['question']
        entity = entry['entity_token']
        question_id = entry['question_id']
        answer = entry['answer']
        img_id = str(entry['image_id'])
        stat_features = torch.from_numpy(np.array(self.stat_features[img_id]))
        # attr_features = torch.from_numpy(np.array(self.attr_features[img_id]))
        # attr_features = {}  # 暂时用不到这个
        ans = entry['ans_token']

        #XVLM
        # ann = self.ann[index]
        #根据question_id来查找对应的XVLM数据
        ann = self.ann_dict.get(question_id)
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

        image = Image.open(image_path).convert('RGB')  # 打开图像文件，并将其转换为RGB格式
        # 对图像进行预处理（例如缩放、归一化等），使用初始化时传入的transform
        image = self.transform(image)
        question_xvlm = pre_question(ann['question'], self.max_ques_words)

        #ISubGVQA
        sg_datum = self.sg_feature_lookup.query_and_translate(img_id)
        sg_datum.x = sg_datum.x.squeeze()
        sg_datum.edge_attr = sg_datum.edge_attr.squeeze()
        scene_graph = sg_datum

        # question_id_xvlm = ann['question_id']

        if answer is not None:  #train
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            # v, b, w, e, q, a, ans, image, question, question_id, scene_graph
            return features, spatials, stat_features, entity, question_ban, target, ans, image, question_xvlm, question_id,scene_graph
        else:   # test
            return features, spatials, stat_features, entity, question_ban, image, question_xvlm, question_id, scene_graph

    def __len__(self):
        return len(self.entries)

