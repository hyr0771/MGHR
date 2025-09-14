"""
优化的批处理函数
替换原来的trim_collate函数，提供更好的性能和错误处理
"""
import logging
import warnings
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import collections

import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric
from torch_geometric.data import Data, Batch

# 配置日志
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 类型映射
int_classes = int
string_classes = str

# NumPy类型映射
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class OptimizedCollateFunction:
    """优化的批处理函数类"""
    
    def __init__(self, use_shared_memory: bool = True, max_boxes: int = 50):
        self.use_shared_memory = use_shared_memory
        self.max_boxes = max_boxes
        self.error_count = 0
        self.max_errors = 10
    
    def __call__(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """主要的批处理函数"""
        try:
            return self._process_batch(batch)
        except Exception as e:
            self.error_count += 1
            logger.error(f"批处理失败 (错误 #{self.error_count}): {e}")
            
            if self.error_count > self.max_errors:
                logger.critical("批处理错误过多，使用简化模式")
                return self._fallback_collate(batch)
            
            return self._safe_collate(batch)
    
    def _process_batch(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """处理批数据"""
        if not batch:
            return {}
        
        # 检查数据类型
        elem_type = type(batch[0])
        
        # 处理tensor类型
        if torch.is_tensor(batch[0]):
            return self._handle_tensor_batch(batch)
        
        # 处理numpy数组
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
            return self._handle_numpy_batch(batch)
        
        # 处理基本数据类型
        elif isinstance(batch[0], (int, float, str)):
            return self._handle_basic_types(batch)
        
        # 处理字典类型
        elif isinstance(batch[0], collections.abc.Mapping):
            return self._handle_dict_batch(batch)
        
        # 处理序列类型
        elif isinstance(batch[0], collections.abc.Sequence):
            return self._handle_sequence_batch(batch)
        
        # 处理torch_geometric数据
        elif isinstance(batch[0], torch_geometric.data.Data):
            return self._handle_geometric_batch(batch)
        
        else:
            raise TypeError(f"不支持的数据类型: {elem_type}")
    
    def _handle_tensor_batch(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """处理tensor批数据"""
        first_tensor = batch[0]
        
        # 3维tensor (图像数据)
        if first_tensor.dim() == 3:
            return {'data': torch.stack(batch, 0)}
        
        # 2维tensor (特征数据)
        elif first_tensor.dim() == 2:
            return {'data': self._pad_and_stack_features(batch)}
        
        # 1维tensor
        else:
            return {'data': torch.stack(batch, 0)}
    
    def _handle_numpy_batch(self, batch: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        """处理numpy数组批数据"""
        elem = batch[0]
        
        # 检查数据类型
        if re.search('[SaUO]', elem.dtype.str) is not None:
            raise TypeError(f"不支持的数据类型: {elem.dtype}")
        
        # 转换为tensor并堆叠
        tensors = [torch.from_numpy(b) for b in batch]
        return {'data': torch.stack(tensors, 0)}
    
    def _handle_basic_types(self, batch: List[Union[int, float, str]]) -> Dict[str, torch.Tensor]:
        """处理基本数据类型"""
        if isinstance(batch[0], int):
            return {'data': torch.LongTensor(batch)}
        elif isinstance(batch[0], float):
            return {'data': torch.DoubleTensor(batch)}
        elif isinstance(batch[0], str):
            return {'data': batch}  # 保持字符串列表
        else:
            raise TypeError(f"不支持的基本数据类型: {type(batch[0])}")
    
    def _handle_dict_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """处理字典批数据"""
        result = {}
        
        # 获取所有键
        keys = batch[0].keys()
        
        for key in keys:
            try:
                # 递归处理每个键的值
                values = [d[key] for d in batch]
                result[key] = self(values)
            except Exception as e:
                logger.warning(f"处理键 '{key}' 时出错: {e}")
                # 跳过这个键
                continue
        
        return result
    
    def _handle_sequence_batch(self, batch: List[collections.abc.Sequence]) -> Dict[str, torch.Tensor]:
        """处理序列批数据"""
        # 转置序列
        transposed = zip(*batch)
        
        # 递归处理每个位置的数据
        result = []
        for samples in transposed:
            try:
                processed = self(samples)
                result.append(processed)
            except Exception as e:
                logger.warning(f"处理序列元素时出错: {e}")
                continue
        
        return {'data': result}
    
    def _handle_geometric_batch(self, batch: List[Data]) -> Dict[str, torch.Tensor]:
        """处理torch_geometric数据批"""
        try:
            return {'data': Batch.from_data_list(batch)}
        except Exception as e:
            logger.error(f"处理几何数据批时出错: {e}")
            return {'data': batch}
    
    def _pad_and_stack_features(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """填充并堆叠特征数据"""
        if not features_list:
            return torch.empty(0)
        
        # 找出最大box数量
        max_num_boxes = max([x.size(0) for x in features_list])
        max_num_boxes = min(max_num_boxes, self.max_boxes)
        
        # 使用共享内存（如果在后台进程中）
        if self.use_shared_memory and torch.utils.data.get_worker_info() is not None:
            numel = len(features_list) * max_num_boxes * features_list[0].size(-1)
            storage = features_list[0].storage()._new_shared(numel)
            out = features_list[0].new(storage)
        else:
            out = None
        
        # 填充并堆叠
        padded_features = []
        for x in features_list:
            # 限制到最大box数量
            x = x[:max_num_boxes]
            # 填充到统一大小
            padded = F.pad(x, (0, 0, 0, max_num_boxes - x.size(0)))
            padded_features.append(padded)
        
        return torch.stack(padded_features, 0, out=out)
    
    def _safe_collate(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """安全的批处理函数"""
        try:
            # 尝试使用默认的collate函数
            return {'data': torch.utils.data.dataloader.default_collate(batch)}
        except Exception as e:
            logger.error(f"默认collate也失败: {e}")
            return self._fallback_collate(batch)
    
    def _fallback_collate(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """回退批处理函数"""
        logger.warning("使用回退批处理函数")
        
        # 返回最简单的批处理结果
        if not batch:
            return {}
        
        # 尝试转换为tensor
        try:
            if torch.is_tensor(batch[0]):
                return {'data': torch.stack(batch, 0)}
            else:
                return {'data': batch}
        except Exception as e:
            logger.error(f"回退处理也失败: {e}")
            return {'data': batch}


# 注意：以下函数已不再使用，主要使用compatible_collate_fn


# 性能监控装饰器
def monitor_performance(func):
    """监控函数性能的装饰器"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"{func.__name__} 执行时间: {end_time - start_time:.4f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} 执行失败 (耗时: {end_time - start_time:.4f}秒): {e}")
            raise
    
    return wrapper


# 兼容原有数据集格式的批处理函数
def compatible_collate_fn(batch):
    """
    兼容原有数据集格式的优化批处理函数
    处理返回tuple格式的数据: (features, spatials, stat_features, entity, question_ban, target, ans, image, question_xvlm, question_id, scene_graph)
    """
    if not batch:
        return []
    
    try:
        # 检查是否为训练模式 (有target) 还是测试模式 (无target)
        first_item = batch[0]
        is_training = len(first_item) == 11  # 训练模式有11个元素
        
        if is_training:
            # 训练模式: features, spatials, stat_features, entity, question_ban, target, ans, image, question_xvlm, question_id, scene_graph
            features_list = [item[0] for item in batch]
            spatials_list = [item[1] for item in batch]
            stat_features_list = [item[2] for item in batch]
            entity_list = [item[3] for item in batch]
            question_ban_list = [item[4] for item in batch]
            target_list = [item[5] for item in batch]
            ans_list = [item[6] for item in batch]
            image_list = [item[7] for item in batch]
            question_xvlm_list = [item[8] for item in batch]
            question_id_list = [item[9] for item in batch]
            scene_graph_list = [item[10] for item in batch]
        else:
            # 测试模式: features, spatials, stat_features, entity, question_ban, image, question_xvlm, question_id, scene_graph
            features_list = [item[0] for item in batch]
            spatials_list = [item[1] for item in batch]
            stat_features_list = [item[2] for item in batch]
            entity_list = [item[3] for item in batch]
            question_ban_list = [item[4] for item in batch]
            image_list = [item[5] for item in batch]
            question_xvlm_list = [item[6] for item in batch]
            question_id_list = [item[7] for item in batch]
            scene_graph_list = [item[8] for item in batch]
            target_list = None
            ans_list = None
        
        # 使用优化的填充和堆叠方法
        collate_fn = OptimizedCollateFunction(use_shared_memory=True, max_boxes=50)
        
        # 处理特征数据 (动态填充)
        features = collate_fn._pad_and_stack_features(features_list)
        spatials = collate_fn._pad_and_stack_features(spatials_list)
        
        # 处理其他数据
        stat_features = torch.stack(stat_features_list) if stat_features_list[0] is not None else None
        entity = torch.stack(entity_list) if entity_list[0] is not None else None
        question_ban = torch.stack(question_ban_list) if question_ban_list[0] is not None else None
        image = torch.stack(image_list) if image_list[0] is not None else None
        
        # 处理可选数据
        target = torch.stack(target_list) if target_list is not None and target_list[0] is not None else None
        ans = torch.stack(ans_list) if ans_list is not None and ans_list[0] is not None else None
        
        # 处理场景图数据
        scene_graph = None
        if scene_graph_list and scene_graph_list[0] is not None:
            try:
                scene_graph = Batch.from_data_list(scene_graph_list)
            except Exception as e:
                logger.warning(f"场景图批处理失败: {e}")
                scene_graph = scene_graph_list  # 回退到原始列表
        
        # 按原有格式返回
        if is_training:
            return features, spatials, stat_features, entity, question_ban, target, ans, image, question_xvlm_list, question_id_list, scene_graph
        else:
            return features, spatials, stat_features, entity, question_ban, image, question_xvlm_list, question_id_list, scene_graph
            
    except Exception as e:
        logger.error(f"兼容批处理失败: {e}")
        # 回退到简单的默认处理
        try:
            import src.utils as utils
            return utils.trim_collate(batch)
        except Exception as fallback_error:
            logger.error(f"回退处理也失败: {fallback_error}")
            # 最后的回退：返回原始batch
            return batch


# 性能监控版本的兼容批处理函数
@monitor_performance
def monitored_compatible_collate_fn(batch):
    """带性能监控的兼容批处理函数"""
    return compatible_collate_fn(batch)


# 导出主要函数
__all__ = [
    'compatible_collate_fn',
    'monitored_compatible_collate_fn'
] 