import os
import copy
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import Dataset, Subset, DataLoader


def get_federated_data(dataset: Union[Dataset, Dict[str, Dataset]], 
                      num_clients: int,
                      target_client_idx: int,
                      forget_dataset: Optional[Dataset] = None) -> Dict[int, Dataset]:
    """
    将数据集分配给多个客户端，确保目标客户端接收遗忘数据

    Args:
        dataset: 主数据集或包含'retain'和'forget'键的字典
        num_clients: 客户端数量
        target_client_idx: 目标客户端索引（接收遗忘数据的客户端）
        forget_dataset: 可选的遗忘数据集，当dataset不是字典时使用

    Returns:
        Dict[int, Dataset]: 键为客户端索引，值为对应的数据集
    """
    if isinstance(dataset, dict):
        # 如果是字典格式，应该包含'retain'和'forget'键
        if 'retain' not in dataset or 'forget' not in dataset:
            raise ValueError("当提供字典类型数据集时，必须包含'retain'和'forget'键")
        retain_dataset = dataset['retain']
        forget_dataset = dataset['forget']
    else:
        # 如果不是字典，则需要提供单独的forget_dataset
        if forget_dataset is None:
            raise ValueError("当提供非字典类型数据集时，必须提供forget_dataset参数")
        retain_dataset = dataset

    # 检查目标客户端索引是否有效
    if target_client_idx < 0 or target_client_idx >= num_clients:
        raise ValueError(f"目标客户端索引必须在0到{num_clients-1}之间")

    # 均匀分割保留数据集
    num_samples = len(retain_dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    # 计算每个客户端应获得的样本数量
    samples_per_client = num_samples // num_clients
    
    # 分配数据给各客户端
    federated_data = {}
    for client_idx in range(num_clients):
        start_idx = client_idx * samples_per_client
        end_idx = start_idx + samples_per_client if client_idx < num_clients - 1 else num_samples
        client_indices = indices[start_idx:end_idx]
        
        # 为每个客户端创建数据子集
        client_dataset = Subset(retain_dataset, client_indices)
        federated_data[client_idx] = client_dataset
    
    # 将遗忘数据分配给目标客户端
    if len(forget_dataset) > 0:
        # 如果目标客户端已有数据，创建一个组合数据集
        class ConcatDataset(Dataset):
            def __init__(self, datasets):
                self.datasets = datasets
                self.lengths = [len(d) for d in datasets]
                self.cumulative_lengths = np.cumsum(self.lengths)
            
            def __len__(self):
                return sum(self.lengths)
            
            def __getitem__(self, idx):
                dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
                if dataset_idx == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - self.cumulative_lengths[dataset_idx - 1]
                return self.datasets[dataset_idx][sample_idx]
        
        federated_data[target_client_idx] = ConcatDataset([federated_data[target_client_idx], forget_dataset])
    
    return federated_data


def fedavg_aggregate(client_models: List[torch.nn.Module], 
                     global_model: torch.nn.Module,
                     weights: Optional[List[float]] = None) -> torch.nn.Module:
    """
    使用联邦平均(FedAvg)算法聚合客户端模型

    Args:
        client_models: 客户端模型列表
        global_model: 全局模型用于存储聚合结果
        weights: 可选的权重列表，用于加权平均

    Returns:
        聚合后的全局模型
    """
    if weights is None:
        # 如果未提供权重，则使用相等权重
        weights = [1.0 / len(client_models) for _ in range(len(client_models))]
    
    # 确保权重和为1
    weights = [w / sum(weights) for w in weights]
    
    # 深度复制全局模型以存储聚合后的结果
    aggregated_model = copy.deepcopy(global_model)
    
    # 获取每个模型的参数字典
    global_dict = aggregated_model.state_dict()
    
    # 对每个参数执行加权平均
    for k in global_dict.keys():
        global_dict[k] = torch.zeros_like(global_dict[k])
        for i, client_model in enumerate(client_models):
            global_dict[k] += client_model.state_dict()[k] * weights[i]
    
    # 将聚合后的参数加载到全局模型
    aggregated_model.load_state_dict(global_dict)
    return aggregated_model


def get_model_difference(model_a: torch.nn.Module, 
                         model_b: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    计算两个模型之间的参数差异

    Args:
        model_a: 第一个模型
        model_b: 第二个模型

    Returns:
        包含参数差异的字典
    """
    diff_dict = {}
    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert name_a == name_b, f"模型参数名称不匹配: {name_a} vs {name_b}"
        diff_dict[name_a] = param_a.data - param_b.data
    return diff_dict


def apply_model_difference(model: torch.nn.Module, 
                           diff_dict: Dict[str, torch.Tensor],
                           scale: float = 1.0) -> torch.nn.Module:
    """
    将差异应用到模型上

    Args:
        model: 要应用差异的模型
        diff_dict: 包含参数差异的字典
        scale: 缩放因子，用于控制应用差异的程度

    Returns:
        应用差异后的模型
    """
    updated_model = copy.deepcopy(model)
    model_dict = updated_model.state_dict()
    
    for name, diff in diff_dict.items():
        if name in model_dict:
            model_dict[name] = model_dict[name] + diff * scale
    
    updated_model.load_state_dict(model_dict)
    return updated_model


def apply_differential_privacy(model: torch.nn.Module, 
                              noise_multiplier: float, 
                              max_grad_norm: float) -> torch.nn.Module:
    """
    对模型参数应用差分隐私
    
    Args:
        model: 要应用差分隐私的模型
        noise_multiplier: 噪声乘数
        max_grad_norm: 最大梯度范数
        
    Returns:
        应用差分隐私后的模型
    """
    dp_model = copy.deepcopy(model)
    model_state = dp_model.state_dict()
    
    # 对模型参数添加噪声
    for name, param in model_state.items():
        if param.requires_grad:
            # 计算噪声标准差
            noise_std = noise_multiplier * max_grad_norm
            # 生成噪声
            noise = torch.randn_like(param) * noise_std
            # 应用噪声
            model_state[name] = param + noise
    
    dp_model.load_state_dict(model_state)
    return dp_model 