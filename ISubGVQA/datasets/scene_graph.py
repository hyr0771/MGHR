from torchtext.data.utils import get_tokenizer
import json
from torchtext.vocab import GloVe, vocab
import numpy as np
import torch
import torch_geometric
import os

from pathlib import Path

current_dir = Path(__file__).parent

class GQASceneGraphs:
    """
    A class to handle GQA Scene Graphs for Visual Question Answering (VQA).
    Attributes:
    -----------
    tokenizer : spacy.tokenizer
        Tokenizer for processing text data.
    vocab_sg : torchtext.vocab.Vocab
        Vocabulary for scene graph encoding.
    vectors : torch.Tensor
        Pre-trained GloVe vectors for the vocabulary.
    scene_graphs_train : dict
        Scene graphs for the training set.
    scene_graphs_valid : dict
        Scene graphs for the validation set.
    scene_graphs_testdev : dict
        Scene graphs for the test development set.
    scene_graphs : dict
        Combined scene graphs from training, validation, and test development sets.
    rel_mapping : dict
        Mapping for relationships.
    obj_mapping : dict
        Mapping for objects.
    attr_mapping : dict
        Mapping for attributes.
    Methods:
    --------
    __init__():
        Initializes the GQASceneGraphs object, builds the vocabulary, and loads scene graphs.
    query_and_translate(queryID: str):
        Queries and translates a scene graph based on the given query ID.
    build_scene_graph_encoding_vocab():
        Builds the vocabulary for scene graph encoding using pre-defined text lists and GloVe vectors.
    convert_one_gqa_scene_graph(sg_this: dict):
        Converts a single GQA scene graph into a PyTorch Geometric data format.
    """
    """
    属性：
        tokenizer : spacy.tokenizer
        用于处理文本数据的分词器。
        
        vocab_sg : torchtext.vocab.Vocab
        用于场景图编码的词汇表。
        
        vectors : torch.Tensor
        用于词汇表的预训练 GloVe 向量。
        
        scene_graphs_train : dict
        训练集的场景图。
        
        scene_graphs_valid : dict
        验证集的场景图。
        
        scene_graphs_testdev : dict
        测试开发集的场景图。
        
        scene_graphs : dict
        从训练集、验证集和测试开发集中合并的场景图。
        
        rel_mapping : dict
        用于关系的映射。
        
        obj_mapping : dict
        用于对象的映射。
        
        attr_mapping : dict
        用于属性的映射。
    
    方法：
        __init__()
        初始化 GQASceneGraphs 对象，构建词汇表，并加载场景图。
        
        query_and_translate(queryID: str)
        根据给定的查询 ID 查询并转换场景图。
        
        build_scene_graph_encoding_vocab()
        使用预定义的文本列表和 GloVe 向量构建场景图编码所需的词汇表。
        
        convert_one_gqa_scene_graph(sg_this: dict)
        将单个 GQA 场景图转换为 PyTorch Geometric 数据格式。
    """
    # tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    def __init__(self):

        self.vocab_sg, self.vectors = self.build_scene_graph_encoding_vocab()
        print(f"Scene graph vocab size: {len(self.vocab_sg)}")

        self.scene_graphs_train = json.load(
            open(str(current_dir.parent.parent) + "/data/gqa/sceneGraphs/train_sceneGraphs.json")
        )
        self.scene_graphs_valid = json.load(
            open(str(current_dir.parent.parent) + "/data/gqa/sceneGraphs/val_sceneGraphs.json")
        )
        self.scene_graphs_test = json.load(
            open(str(current_dir.parent.parent) + "/data/gqa/sceneGraphs/test_sceneGraphs.json")
        )

        # self.scene_graphs = (
        #     self.scene_graphs_train
        #     | self.scene_graphs_valid
        # )
        self.scene_graphs = {**self.scene_graphs_test,**self.scene_graphs_train, **self.scene_graphs_valid}  #重复键的值以后者为准

        self.rel_mapping = {}
        self.obj_mapping = {}
        self.attr_mapping = {}

    '''
    query_and_translate
    传入的数据
        参数：queryID（字符串）
        
        作用：作为唯一标识符，用于从预加载的场景图数据（self.scene_graphs）中查找对应的场景图。
        
        示例：
        
        若queryID为"n30000"，则尝试从self.scene_graphs中获取键为"n30000"的场景图。
        
        若不存在，则使用默认的empty_sg（包含虚拟节点和边）。
    返回的数据
        类型：torch_geometric.data.Data 对象
        包含内容：
        
        节点特征（x）：
            每个节点的特征由 对象名称 和 属性 组成，通过词汇表映射为索引。
            形状：(num_nodes, 4)，其中 4 表示 [对象名称, 属性1, 属性2, 属性3]，不足部分用 <pad> 填充。
        边索引（edge_index）：
            表示图中节点之间的连接关系，格式为 [[源节点索引], [目标节点索引]]。
            包含 双向边（确保对称性）和 自环边（<self>）。
        边属性（edge_attr）：
            每条边的特征由 关系名称 映射为索引。
            形状：(num_edges, 1)。
        附加信息：
            x_bbox：节点的边界框坐标（[x1, y1, x2, y2]），用于空间信息编码。
            added_sym_edge：记录因对称性新增的边的索引。
    假设 queryID 对应的场景图如下：
    {
      "objects": {
        "0": {"name": "car", "relations": [{"object": "1", "name": "near"}], "attributes": ["red"]},
        "1": {"name": "tree", "relations": [{"object": "0", "name": "near"}], "attributes": ["tall"]}
      }
    }
    转换后：
    节点特征：
        节点0：[car_idx, red_idx, <pad>, <pad>]
        节点1：[tree_idx, tall_idx, <pad>, <pad>]
    边索引：
        [[0, 0], [0, 1], [1, 1], [1, 0]]（自环边 + 双向边）
    边属性：
        [<self>_idx, near_idx, <self>_idx, near_idx]


    '''
    def query_and_translate(self, queryID: str):
        empty_sg = {
            "objects": {
                "0": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "1",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "1": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "0",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "2": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "3",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "3": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "1",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "4": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "5",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "5": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "3",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
            }
        }
        sg_this = self.scene_graphs.get(queryID, empty_sg)
        sg_datum = self.convert_one_gqa_scene_graph(sg_this)
        if sg_datum.edge_index.size(1) == 1:
            sg_datum = self.convert_one_gqa_scene_graph(empty_sg)

        return sg_datum

    def build_scene_graph_encoding_vocab(self):
        def load_str_list(fname):
            with open(fname) as f:
                lines = f.read().splitlines()
            return lines

        # current_dir = Path(__file__).parent

        tmp_text_list = []
        tmp_text_list += load_str_list(str(current_dir.parent.parent) + "/data/gqa/meta_info/name_gqa.txt")
        tmp_text_list += load_str_list(str(current_dir.parent.parent) + "/data/gqa/meta_info/attr_gqa.txt")
        tmp_text_list += load_str_list(str(current_dir.parent.parent) + "/data/gqa/meta_info/rel_gqa.txt")

        objects_inv = json.load(open(str(current_dir.parent.parent) + "/data/gqa/meta_info/objects.json"))
        relations_inv = json.load(open(str(current_dir.parent.parent) + "/data/gqa/meta_info/predicates.json"))
        attributes_inv = json.load(open(str(current_dir.parent.parent) + "/data/gqa/meta_info/attributes.json"))

        only_test_object = json.load(open(str(current_dir.parent.parent) + "/data/gqa/meta_info/gqa_only_test_object_names.json"))
        only_test_relation = json.load(open(str(current_dir.parent.parent) + "/data/gqa/meta_info/gqa_only_test_relation_names.json"))

        tmp_text_list += objects_inv + relations_inv + attributes_inv + only_test_object + only_test_relation
        tmp_text_list.append("<self>")
        tmp_text_list.append("pokemon")  # add special token for self-connection
        tmp_text_list = list(dict.fromkeys(tmp_text_list))  #去重，减少运算
        tmp_text_list = [tmp_text_list]

        sg_vocab_stoi = {token: i for i, token in enumerate(tmp_text_list[0])}
        if os.path.exists(str(current_dir.parent.parent) + "/data/gqa/vocabs/sg_vocab.pt"):
            print("loading scene graph vocab...")
            sg_vocab = torch.load(str(current_dir.parent.parent) + "/data/gqa/vocabs/sg_vocab.pt")
        else:
            print("creating scene graph vocab...")
            sg_vocab = vocab(
                sg_vocab_stoi,
                specials=[
                    "<unk>",
                    "<pad>",
                    "<sos>",
                    "<eos>",
                    "<self>",
                ],
            )
            print("saving text vocab...")
            torch.save(sg_vocab, str(current_dir.parent.parent) + "/data/gqa/vocabs/sg_vocab.pt")

        myvec = GloVe(name="6B", dim=300)
        vectors = torch.randn((len(sg_vocab.vocab.itos_), 300))

        for i, token in enumerate(sg_vocab.vocab.itos_):
            glove_idx = myvec.stoi.get(token)
            if glove_idx:
                vectors[i] = myvec.vectors[glove_idx]

        assert torch.all(
            #若转换后的图仅有一条边（如 edge_index.size(1) == 1），说明原始场景图可能无效或为空。
            # 此时强制使用 empty_sg 生成默认数据，确保输出的 Data 对象格式完整，避免后续模型处理出错。
            myvec.vectors[myvec.stoi.get("helmet")]
            == vectors[sg_vocab.vocab.get_stoi()["helmet"]]
        )
        return sg_vocab, vectors

    '''
    传入的数据
        参数：sg_this（字典）
        结构：
        {
          "objects": {
            "0": {  以对象ID为键，每个对象包含以下字段
              "name": "car",
              "relations": [{"object": "1", "name": "near"}],   每个关系包含"object"（目标对象ID）和"name"（关系名称，如"near")
              "attributes": ["red", "shiny"]    属性列表
              （可选）"x", "y", "w", "h"：边界框坐标。
            },
            "1": {
              "name": "tree",
              "relations": [{"object": "0", "name": "near"}],
              "attributes": ["tall"]
            }
          }
        }
    返回的数据
        类型：torch_geometric.data.Data 对象
        包含内容：
            节点特征（x）：
                形状：(num_nodes, 4)，每个节点的特征由 对象名称 + 最多3个属性 组成，映射为词汇表索引。
                填充规则：不足部分用<pad>（索引为self.vocab_sg.get_stoi().get("<pad>")）填充。
                示例：
                    节点0：[car_idx, red_idx, shiny_idx, <pad>]
            边索引（edge_index）：
                形状：(2, num_edges)，表示边的拓扑结构（[[源节点索引], [目标节点索引]]）。
                包含：
                    自环边（如[0, 0]），边属性为<self>。
                    双向边（如[0, 1]和[1, 0]），确保关系对称。
            边属性（edge_attr）：
                形状：(num_edges, 1)，每条边的特征为 关系名称 的词汇表索引。
            附加字段：
                x_bbox：形状(num_nodes, 4)，记录每个对象的边界框坐标（[x1, y1, x2, y2]）。
                added_sym_edge：形状(num_added_edges,)，记录因对称性新增的边的索引（例如反向边的位置）。  
    
    针对上面传入的数据，经过convert_one_gqa_scene_graph函数之后得出的结果
    Data(
      x=torch.Tensor([[car_idx, red_idx, shiny_idx, <pad>], 
                     [tree_idx, tall_idx, <pad>, <pad>]]),  # shape (2,4)  (num_nodes, 4)
      edge_index=torch.Tensor([[0, 0, 0, 1, 1, 1], 
                              [0, 1, 0, 1, 0, 1]]),         # shape (2,6)   (2, num_edges)
      edge_attr=torch.Tensor([[<self>_idx], 
                              [near_idx], 
                              [near_idx], 
                              [<self>_idx], 
                              [near_idx], 
                              [near_idx]]),                  # shape (6,1)  (num_edges, 1)
      x_bbox=torch.Tensor([[...], [...]])                    # 边界框坐标
      added_sym_edge=torch.Tensor([2, 4, 5])                 # 新增反向边的索引
    )
    '''
    def convert_one_gqa_scene_graph(self, sg_this):
        # assert len(sg_this['objects'].keys()) != 0, sg_this
        if len(sg_this["objects"].keys()) == 0:
            # only in val
            # print("Got Empty Scene Graph", sg_this) # only one empty scene graph during val
            # use a dummy scene graph instead
            sg_this = {
                "objects": {
                    "0": {
                        "name": "<unk>",
                        "relations": [
                            {
                                "object": "1",
                                "name": "<unk>",
                            }
                        ],
                        "attributes": ["<unk>"],
                    },
                    "1": {
                        "name": "<unk>",
                        "relations": [
                            {
                                "object": "0",
                                "name": "<unk>",
                            }
                        ],
                        "attributes": ["<unk>"],
                    },
                }
            }

        ##################################
        # graph node: objects
        ##################################
        objIDs = sorted(sg_this["objects"].keys())  # str
        map_objID_to_node_idx = {
            objID: node_idx for node_idx, objID in enumerate(objIDs)
        }

        ##################################
        # Initialize Three key components for graph representation
        ##################################
        node_feature_list = []
        edge_feature_list = []
        # [[from, to], ...]
        edge_topology_list = []
        added_sym_edge_list = (
            []
        )  # yanhao: record the index of added edges in the edge_feature_list
        bbox_coordinates = []
        ##################################
        # Duplicate edges, making sure that the topology is symmetric
        ##################################
        from_to_connections_set = set()
        for node_idx in range(len(objIDs)):
            objId = objIDs[node_idx]
            obj = sg_this["objects"][objId]
            for rel in obj["relations"]:
                # [from self as source, to outgoing]
                from_to_connections_set.add(
                    (node_idx, map_objID_to_node_idx[rel["object"]])
                )
        # print("from_to_connections_set", from_to_connections_set)

        for node_idx in range(len(objIDs)):
            ##################################
            # Traverse Scene Graph's objects based on node idx order
            ##################################
            objId = objIDs[node_idx]
            obj = sg_this["objects"][objId]

            ##################################
            # Encode Node Feature: object category, attributes
            # Note: not encoding spatial information
            # - obj['x'], obj['y'], obj['w'], obj['h']
            ##################################
            # MAX_OBJ_TOKEN_LEN = 4 # 1 name + 3 attributes
            MAX_OBJ_TOKEN_LEN = 4

            # 4 X '<pad>'
            object_token_arr = np.ones(
                MAX_OBJ_TOKEN_LEN, dtype=np.int_
            ) * self.vocab_sg.get_stoi().get("<pad>")

            # should have no error
            obj_name = self.obj_mapping.get(obj["name"], obj["name"])
            object_token_arr[0] = self.vocab_sg.get_stoi().get(obj_name, 1)
            # assert object_token_arr[0] !=0 , obj
            if object_token_arr[0] == 0:
                # print("Out Of Vocabulary Object:", obj['name'])
                pass

            counter = 0
            for attr_idx, attr in enumerate(set(obj["attributes"])):
                if counter >= 3:
                    break
                attr = self.attr_mapping.get(attr, attr)
                object_token_arr[attr_idx + 1] = self.vocab_sg.get_stoi().get(attr, 1)
                counter += 1

            obj_bbox = [
                obj.get("x1", -1),
                obj.get("y1", -1),
                obj.get("x2", -1),
                obj.get("y2", -1),
            ]

            node_feature_list.append(object_token_arr)
            bbox_coordinates.append(obj_bbox)

            edge_topology_list.append([node_idx, node_idx])  # [from self, to self]
            edge_token_arr = np.array(
                [self.vocab_sg.get_stoi()["<self>"]], dtype=np.int_
            )
            edge_feature_list.append(edge_token_arr)

            for rel in obj["relations"]:
                # [from self as source, to outgoing]
                edge_topology_list.append(
                    [node_idx, map_objID_to_node_idx[rel["object"]]]
                )
                # name of the relationship
                rel_name = self.rel_mapping.get(rel["name"], rel["name"])

                edge_token_arr = np.array(
                    [self.vocab_sg.get_stoi().get(rel_name, 1)], dtype=np.int_
                )
                edge_feature_list.append(edge_token_arr)

                # Symmetric
                if (
                    map_objID_to_node_idx[rel["object"]],
                    node_idx,
                ) not in from_to_connections_set:
                    # print("catch!", (map_objID_to_node_idx[rel["object"]], node_idx), rel["name"])

                    # reverse of [from self as source, to outgoing]
                    edge_topology_list.append(
                        [map_objID_to_node_idx[rel["object"]], node_idx]
                    )
                    # re-using name of the relationship
                    edge_feature_list.append(edge_token_arr)

                    # yanhao: record the added edge's index in feature and idx array:
                    added_sym_edge_list.append(len(edge_feature_list) - 1)

        ##################################
        # Convert to standard pytorch geometric format
        # - node_feature_list
        # - edge_feature_list
        # - edge_topology_list
        ##################################

        # print("sg_this", sg_this)
        # print("objIDs", objIDs)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)

        node_feature_list_arr = np.stack(node_feature_list, axis=0)
        # print("node_feature_list_arr", node_feature_list_arr.shape)

        obj_bbox_list_arr = np.stack(bbox_coordinates, axis=0)

        edge_feature_list_arr = np.stack(edge_feature_list, axis=0)
        # print("edge_feature_list_arr", edge_feature_list_arr.shape)

        edge_topology_list_arr = np.stack(edge_topology_list, axis=0)
        # print("edge_topology_list_arr", edge_topology_list_arr.shape)
        del edge_topology_list_arr

        # edge_index = torch.tensor([[0, 1],
        #                         [1, 0],
        #                         [1, 2],
        #                         [2, 1]], dtype=torch.long)
        edge_index = torch.tensor(edge_topology_list, dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        x = torch.from_numpy(node_feature_list_arr).long()
        edge_attr = torch.from_numpy(edge_feature_list_arr).long()
        x_bbox = torch.from_numpy(obj_bbox_list_arr)
        datum = torch_geometric.data.Data(
            x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr
        )

        # yanhao: add an additional variable to datum:
        added_sym_edge = torch.LongTensor(added_sym_edge_list)
        datum.added_sym_edge = added_sym_edge

        datum.x_bbox = x_bbox

        return datum



###########
#   VQA Dataet
###########
class VQASceneGraphs:
    def __init__(self):
        self.vocab_sg, self.vectors = self.build_scene_graph_encoding_vocab()
        print(f"Scene graph vocab size: {len(self.vocab_sg)}")

        self.scene_graphs_train = json.load(
            open(str(current_dir.parent.parent) + "/data/vqa/sceneGraphs/train_sceneGraphs.json")
        )
        self.scene_graphs_valid = json.load(
            open(str(current_dir.parent.parent) + "/data/vqa/sceneGraphs/val_sceneGraphs.json")
        )
        self.scene_graphs_test = json.load(
            open(str(current_dir.parent.parent) + "/data/vqa/sceneGraphs/test_sceneGraphs.json")
        )

        self.scene_graphs = {**self.scene_graphs_test,**self.scene_graphs_train, **self.scene_graphs_valid}

        self.rel_mapping = {}
        self.obj_mapping = {}
        self.attr_mapping = {}

    def query_and_translate(self, queryID: str):
        empty_sg = {
            "objects": {
                "0": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "1",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "1": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "0",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "2": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "3",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "3": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "1",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "4": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "5",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
                "5": {
                    "name": "<unk>",
                    "relations": [
                        {
                            "object": "3",
                            "name": "<unk>",
                        }
                    ],
                    "attributes": ["<unk>"],
                },
            }
        }
        sg_this = self.scene_graphs.get(queryID, empty_sg)
        if sg_this is empty_sg:
            print(f"{queryID}没找到！")
        sg_datum = self.convert_one_gqa_scene_graph(sg_this)
        if sg_datum.edge_index.size(1) == 1:
            sg_datum = self.convert_one_gqa_scene_graph(empty_sg)
        return sg_datum

    def build_scene_graph_encoding_vocab(self):
        def load_str_list(fname):
            with open(fname) as f:
                lines = f.read().splitlines()
            return lines

        # current_dir = Path(__file__).parent

        tmp_text_list = []
        tmp_text_list += load_str_list(str(current_dir.parent.parent) + "/data/vqa/meta_info/name_gqa.txt")
        tmp_text_list += load_str_list(str(current_dir.parent.parent) + "/data/vqa/meta_info/attr_gqa.txt")
        tmp_text_list += load_str_list(str(current_dir.parent.parent) + "/data/vqa/meta_info/rel_gqa.txt")

        objects_inv = json.load(open(str(current_dir.parent.parent) + "/data/vqa/meta_info/objects.json"))
        relations_inv = json.load(open(str(current_dir.parent.parent) + "/data/vqa/meta_info/predicates.json"))
        attributes_inv = json.load(open(str(current_dir.parent.parent) + "/data/vqa/meta_info/attributes.json"))

        vqa_SG_attribute = json.load(open(str(current_dir.parent.parent) + "/data/vqa/meta_info/vqa_attribute_names.json"))
        vqa_SG_object = json.load(open(str(current_dir.parent.parent) + "/data/vqa/meta_info/vqa_object_names.json"))
        vqa_SG_relation = json.load(open(str(current_dir.parent.parent) + "/data/vqa/meta_info/vqa_relation_names.json"))

        #


        tmp_text_list += objects_inv + relations_inv + attributes_inv + vqa_SG_attribute + vqa_SG_object + vqa_SG_relation
        tmp_text_list.append("<self>")
        tmp_text_list.append("pokemon")  # add special token for self-connection
        tmp_text_list = list(dict.fromkeys(tmp_text_list))  # 去重，减少运算
        tmp_text_list = [tmp_text_list]

        sg_vocab_stoi = {token: i for i, token in enumerate(tmp_text_list[0])}
        if os.path.exists(str(current_dir.parent.parent) + "/data/vqa/vocabs/sg_vocab.pt"):
            print("loading scene graph vocab...")
            sg_vocab = torch.load(str(current_dir.parent.parent) + "/data/vqa/vocabs/sg_vocab.pt")
        else:
            print("creating scene graph vocab...")
            sg_vocab = vocab(
                sg_vocab_stoi,
                specials=[
                    "<unk>",
                    "<pad>",
                    "<sos>",
                    "<eos>",
                    "<self>",
                ],
            )
            print("saving text vocab...")
            torch.save(sg_vocab, str(current_dir.parent.parent) + "/data/vqa/vocabs/sg_vocab.pt")

        myvec = GloVe(name="6B", dim=300)
        vectors = torch.randn((len(sg_vocab.vocab.itos_), 300))

        for i, token in enumerate(sg_vocab.vocab.itos_):
            glove_idx = myvec.stoi.get(token)
            if glove_idx:
                vectors[i] = myvec.vectors[glove_idx]

        assert torch.all(
            # 若转换后的图仅有一条边（如 edge_index.size(1) == 1），说明原始场景图可能无效或为空。
            # 此时强制使用 empty_sg 生成默认数据，确保输出的 Data 对象格式完整，避免后续模型处理出错。
            myvec.vectors[myvec.stoi.get("helmet")]
            == vectors[sg_vocab.vocab.get_stoi()["helmet"]]
        )
        return sg_vocab, vectors



    def convert_one_gqa_scene_graph(self, sg_this):
        # assert len(sg_this['objects'].keys()) != 0, sg_this
        if len(sg_this["objects"].keys()) == 0:
            # only in val
            # print("Got Empty Scene Graph", sg_this) # only one empty scene graph during val
            # use a dummy scene graph instead
            sg_this = {
                "objects": {
                    "0": {
                        "name": "<unk>",
                        "relations": [
                            {
                                "object": "1",
                                "name": "<unk>",
                            }
                        ],
                        "attributes": ["<unk>"],
                    },
                    "1": {
                        "name": "<unk>",
                        "relations": [
                            {
                                "object": "0",
                                "name": "<unk>",
                            }
                        ],
                        "attributes": ["<unk>"],
                    },
                }
            }

        ##################################
        # graph node: objects
        ##################################
        objIDs = sorted(sg_this["objects"].keys())  # str
        map_objID_to_node_idx = {
            objID: node_idx for node_idx, objID in enumerate(objIDs)
        }

        ##################################
        # Initialize Three key components for graph representation
        ##################################
        node_feature_list = []
        edge_feature_list = []
        # [[from, to], ...]
        edge_topology_list = []
        added_sym_edge_list = (
            []
        )  # yanhao: record the index of added edges in the edge_feature_list
        bbox_coordinates = []
        ##################################
        # Duplicate edges, making sure that the topology is symmetric
        ##################################
        from_to_connections_set = set()
        for node_idx in range(len(objIDs)):
            objId = objIDs[node_idx]
            obj = sg_this["objects"][objId]
            for rel in obj["relations"]:
                # [from self as source, to outgoing]
                from_to_connections_set.add(
                    (node_idx, map_objID_to_node_idx[rel["object"]])
                )
        # print("from_to_connections_set", from_to_connections_set)

        for node_idx in range(len(objIDs)):
            ##################################
            # Traverse Scene Graph's objects based on node idx order
            ##################################
            objId = objIDs[node_idx]
            obj = sg_this["objects"][objId]

            ##################################
            # Encode Node Feature: object category, attributes
            # Note: not encoding spatial information
            # - obj['x'], obj['y'], obj['w'], obj['h']
            ##################################
            # MAX_OBJ_TOKEN_LEN = 4 # 1 name + 3 attributes
            MAX_OBJ_TOKEN_LEN = 4

            # 4 X '<pad>'
            object_token_arr = np.ones(
                MAX_OBJ_TOKEN_LEN, dtype=np.int_
            ) * self.vocab_sg.get_stoi().get("<pad>")

            # should have no error
            obj_name = self.obj_mapping.get(obj["name"], obj["name"])
            object_token_arr[0] = self.vocab_sg.get_stoi().get(obj_name, 1)
            # assert object_token_arr[0] !=0 , obj
            if object_token_arr[0] == 0:
                # print("Out Of Vocabulary Object:", obj['name'])
                pass

            counter = 0
            for attr_idx, attr in enumerate(set(obj["attributes"])):
                if counter >= 3:
                    break
                attr = self.attr_mapping.get(attr, attr)
                object_token_arr[attr_idx + 1] = self.vocab_sg.get_stoi().get(attr, 1)
                counter += 1

            obj_bbox = [
                obj.get("x1", -1),
                obj.get("y1", -1),
                obj.get("x2", -1),
                obj.get("y2", -1),
            ]

            node_feature_list.append(object_token_arr)
            bbox_coordinates.append(obj_bbox)

            edge_topology_list.append([node_idx, node_idx])  # [from self, to self]
            edge_token_arr = np.array(
                [self.vocab_sg.get_stoi()["<self>"]], dtype=np.int_
            )
            edge_feature_list.append(edge_token_arr)

            for rel in obj["relations"]:
                # [from self as source, to outgoing]
                edge_topology_list.append(
                    [node_idx, map_objID_to_node_idx[rel["object"]]]
                )
                # name of the relationship
                rel_name = self.rel_mapping.get(rel["name"], rel["name"])

                edge_token_arr = np.array(
                    [self.vocab_sg.get_stoi().get(rel_name, 1)], dtype=np.int_
                )
                edge_feature_list.append(edge_token_arr)

                # Symmetric
                if (
                        map_objID_to_node_idx[rel["object"]],
                        node_idx,
                ) not in from_to_connections_set:
                    # print("catch!", (map_objID_to_node_idx[rel["object"]], node_idx), rel["name"])

                    # reverse of [from self as source, to outgoing]
                    edge_topology_list.append(
                        [map_objID_to_node_idx[rel["object"]], node_idx]
                    )
                    # re-using name of the relationship
                    edge_feature_list.append(edge_token_arr)

                    # yanhao: record the added edge's index in feature and idx array:
                    added_sym_edge_list.append(len(edge_feature_list) - 1)

        ##################################
        # Convert to standard pytorch geometric format
        # - node_feature_list
        # - edge_feature_list
        # - edge_topology_list
        ##################################

        # print("sg_this", sg_this)
        # print("objIDs", objIDs)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)

        node_feature_list_arr = np.stack(node_feature_list, axis=0)
        # print("node_feature_list_arr", node_feature_list_arr.shape)

        obj_bbox_list_arr = np.stack(bbox_coordinates, axis=0)

        edge_feature_list_arr = np.stack(edge_feature_list, axis=0)
        # print("edge_feature_list_arr", edge_feature_list_arr.shape)

        edge_topology_list_arr = np.stack(edge_topology_list, axis=0)
        # print("edge_topology_list_arr", edge_topology_list_arr.shape)
        del edge_topology_list_arr

        # edge_index = torch.tensor([[0, 1],
        #                         [1, 0],
        #                         [1, 2],
        #                         [2, 1]], dtype=torch.long)
        edge_index = torch.tensor(edge_topology_list, dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        x = torch.from_numpy(node_feature_list_arr).long()
        edge_attr = torch.from_numpy(edge_feature_list_arr).long()
        x_bbox = torch.from_numpy(obj_bbox_list_arr)
        datum = torch_geometric.data.Data(
            x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr
        )

        # yanhao: add an additional variable to datum:
        added_sym_edge = torch.LongTensor(added_sym_edge_list)
        datum.added_sym_edge = added_sym_edge

        datum.x_bbox = x_bbox

        return datum