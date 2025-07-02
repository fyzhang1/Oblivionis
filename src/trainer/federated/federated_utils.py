import torch
import logging
import logging
import copy
from typing import List, Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def FedAvg(client_state_dicts, global_model_state_dict=None):
    """
    执行FedAvg策略的模型参数聚合
    
    Args:
        client_state_dicts: 客户端模型状态字典列表
        global_model_state_dict: 全局模型状态字典，用于处理空参数情况
        
    Returns:
        全局聚合后的状态字典
    """
    global_state_dict = {}
    for key in client_state_dicts[0].keys():
        if isinstance(client_state_dicts[0][key], torch.Tensor):
            valid_params = [state_dict[key] for state_dict in client_state_dicts if state_dict[key].numel() > 0]
            if valid_params:
                global_state_dict[key] = torch.stack(valid_params).mean(dim=0)
            else:
                if global_model_state_dict is not None:
                    global_state_dict[key] = global_model_state_dict[key]
                    logger.warning(f"All clients have empty parameter for {key}, using global model parameter")
                else:
                    raise ValueError(f"All clients have empty parameter for {key} and no global model provided")
        else:
            global_state_dict[key] = client_state_dicts[0][key]
    
    return global_state_dict


def FedProx(client_state_dicts, global_model_state_dict=None, mu=0.01):
    """
    
    Args:
        client_state_dicts: 客户端模型状态字典列表
        global_model_state_dict: 全局模型状态字典
        mu: 正则化系数，仅用于记录
        
    Returns:
        全局聚合后的状态字典
    """
    # FedProx的聚合方式与FedAvg相同，区别在于客户端训练阶段
    return FedAvg(client_state_dicts, global_model_state_dict)


def FedAvgM(client_state_dicts: List[Dict], global_model_state_dict: Dict,
                        server_momentum: Dict = None, momentum_factor: float = 0.9) -> Tuple[Dict, Dict]:
    """
    执行FedAvgM策略的模型参数聚合 - 带动量的联邦平均
    
    Args:
        client_state_dicts: 客户端模型状态字典列表
        global_model_state_dict: 全局模型状态字典
        server_momentum: 服务器动量状态字典
        momentum_factor: 动量因子
        
    Returns:
        (全局聚合后的状态字典, 更新后的服务器动量)
    """
    # 1. 计算当前轮次的模型参数平均值
    avg_state_dict = FedAvg(client_state_dicts, global_model_state_dict)
    
    # 2. 初始化动量状态字典
    if server_momentum is None:
        server_momentum = {}
        for key, param in global_model_state_dict.items():
            if isinstance(param, torch.Tensor):
                server_momentum[key] = torch.zeros_like(param)
            else:
                server_momentum[key] = param
    
    # 3. 应用动量更新
    new_global_state_dict = {}
    new_server_momentum = {}
    
    for key in global_model_state_dict.keys():
        if isinstance(global_model_state_dict[key], torch.Tensor):
            # 计算参数更新值
            delta = avg_state_dict[key] - global_model_state_dict[key]
            
            # 更新动量
            new_server_momentum[key] = momentum_factor * server_momentum[key] + delta
            
            # 应用动量更新到全局模型
            new_global_state_dict[key] = global_model_state_dict[key] + new_server_momentum[key]
        else:
            new_global_state_dict[key] = avg_state_dict[key]
            new_server_momentum[key] = server_momentum[key]
    
    return new_global_state_dict, new_server_momentum


# def FedAdagrad(client_state_dicts: List[Dict], global_model_state_dict: Dict,
#                           server_velocity: Dict = None, learning_rate: float = 0.1, 
#                           epsilon: float = 1e-8, tau: float = 0.0) -> Tuple[Dict, Dict]:
#     """
#     执行FedAdagrad策略的模型参数聚合 - 自适应学习率的联邦聚合
    
#     Args:
#         client_state_dicts: 客户端模型状态字典列表
#         global_model_state_dict: 全局模型状态字典
#         server_velocity: 服务器累积梯度平方和
#         learning_rate: 学习率
#         epsilon: 数值稳定性参数
#         tau: 初始累积项
        
#     Returns:
#         (全局聚合后的状态字典, 更新后的服务器累积梯度平方和)
#     """
#     # 1. 计算当前轮次的模型参数平均值
#     avg_state_dict = FedAvg(client_state_dicts, global_model_state_dict)
    
#     # 2. 初始化累积梯度平方和
#     if server_velocity is None:
#         server_velocity = {}
#         for key, param in global_model_state_dict.items():
#             if isinstance(param, torch.Tensor):
#                 server_velocity[key] = tau * torch.ones_like(param)
#             else:
#                 server_velocity[key] = param
    
#     # 3. 应用Adagrad更新
#     new_global_state_dict = {}
#     new_server_velocity = {}
    
#     for key in global_model_state_dict.keys():
#         if isinstance(global_model_state_dict[key], torch.Tensor):
#             # 计算参数更新值
#             delta = avg_state_dict[key] - global_model_state_dict[key]
            
#             # 更新累积梯度平方和
#             new_server_velocity[key] = server_velocity[key] + delta.pow(2)
            
#             # 应用自适应学习率更新到全局模型
#             factor = learning_rate / (torch.sqrt(new_server_velocity[key]) + epsilon)
#             new_global_state_dict[key] = global_model_state_dict[key] + factor * delta
#         else:
#             new_global_state_dict[key] = avg_state_dict[key]
#             new_server_velocity[key] = server_velocity[key]
    
#     return new_global_state_dict, new_server_velocity


def FedAdagrad(client_state_dicts: List[Dict], global_model_state_dict: Dict,
                          epsilon: float = 1e-3, tau: float = 1e-3) -> Tuple[Dict, Dict]:
    """
    执行FedAdagrad策略的模型参数聚合 - 自适应学习率的联邦聚合
    
    Args:
        client_state_dicts: 客户端模型状态字典列表
        global_model_state_dict: 全局模型状态字典
        server_velocity: 服务器累积梯度平方和
        learning_rate: 学习率
        epsilon: 数值稳定性参数
        tau: 初始累积项
        
    Returns:
        (全局聚合后的状态字典, 更新后的服务器累积梯度平方和)
    """
    opt_proxy_dict = None
    proxy_dict = None
    proxy_dict, opt_proxy_dict = {}, {}
    num_clients = len(client_state_dicts)
    for key in global_model_state_dict.keys():
        proxy_dict[key] = torch.zeros_like(global_model_state_dict[key])
        opt_proxy_dict[key] = torch.ones_like(global_model_state_dict[key]) * tau**2

    for key, param in opt_proxy_dict.items():
        delta = sum([(client_state_dicts[c][key] - global_model_state_dict[key]) 
                         for c in range(num_clients)]) / num_clients
        proxy_dict[key] = delta
        opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
        global_model_state_dict[key] += epsilon* torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+tau)
    
    return global_model_state_dict


# def FedYogi(client_state_dicts: List[Dict], global_model_state_dict: Dict,
#                        server_velocity: Dict = None, server_momentum: Dict = None,
#                        learning_rate: float = 0.1, beta1: float = 0.9, beta2: float = 0.99,
#                        epsilon: float = 1e-3, tau: float = 0.0) -> Tuple[Dict, Dict, Dict]:
#     # 1. 计算当前轮次的模型参数平均值
#     avg_state_dict = FedAvg(client_state_dicts, global_model_state_dict)
    
#     # 2. 初始化动量和速度
#     if server_momentum is None:
#         server_momentum = {}
#         for key, param in global_model_state_dict.items():
#             if isinstance(param, torch.Tensor):
#                 server_momentum[key] = torch.zeros_like(param)
#             else:
#                 server_momentum[key] = param
    
#     if server_velocity is None:
#         server_velocity = {}
#         for key, param in global_model_state_dict.items():
#             if isinstance(param, torch.Tensor):
#                 server_velocity[key] = tau * torch.ones_like(param)
#             else:
#                 server_velocity[key] = param
    
#     # 3. 应用Yogi更新
#     new_global_state_dict = {}
#     new_server_momentum = {}
#     new_server_velocity = {}
    
#     for key in global_model_state_dict.keys():
#         if isinstance(global_model_state_dict[key], torch.Tensor):
#             # 计算参数更新值
#             delta = avg_state_dict[key] - global_model_state_dict[key]
            
#             # 更新一阶动量
#             new_server_momentum[key] = beta1 * server_momentum[key] + (1 - beta1) * delta
            
#             # 更新二阶动量 (Yogi方式)
#             g_square = delta.pow(2)
#             v_diff = g_square - server_velocity[key]
#             new_server_velocity[key] = server_velocity[key] + (1 - beta2) * torch.sign(v_diff) * v_diff
            
#             # 应用Yogi更新到全局模型
#             adaptive_lr = learning_rate / (torch.sqrt(new_server_velocity[key]) + epsilon)
#             new_global_state_dict[key] = global_model_state_dict[key] + adaptive_lr * new_server_momentum[key]
#         else:
#             new_global_state_dict[key] = avg_state_dict[key]
#             new_server_momentum[key] = server_momentum[key]
#             new_server_velocity[key] = server_velocity[key]
    
#     return new_global_state_dict, new_server_velocity, new_server_momentum


def FedYogi(client_state_dicts: List[Dict], global_model_state_dict: Dict,
                      round_idx: int = 0, beta1: float = 0.9, beta2: float = 0.99,
                        epsilon: float = 1e-3, tau: float = 1e-3) -> Tuple[Dict, Dict, Dict]:
    """
    执行FedYogi策略的模型参数聚合 - Yogi优化器的联邦版本
    
    Args:
        client_state_dicts: 客户端模型状态字典列表
        global_model_state_dict: 全局模型状态字典
        server_velocity: 服务器累积二阶动量
        server_momentum: 服务器一阶动量
        learning_rate: 学习率
        beta1: 一阶动量衰减率
        beta2: 二阶动量衰减率
        epsilon: 数值稳定性参数
        tau: 初始累积项
        
    Returns:
        (全局聚合后的状态字典, 更新后的服务器二阶动量, 更新后的服务器一阶动量)
    """

    opt_proxy_dict = None
    proxy_dict = None
    proxy_dict, opt_proxy_dict = {}, {}
    num_clients = len(client_state_dicts)
    for key in global_model_state_dict.keys():
        proxy_dict[key] = torch.zeros_like(global_model_state_dict[key])
        opt_proxy_dict[key] = torch.ones_like(global_model_state_dict[key]) * tau**2

    
    for key, param in opt_proxy_dict.items():
        delta = sum([(client_state_dicts[c][key] - global_model_state_dict[key]) 
                         for c in range(num_clients)]) / num_clients
        proxy_dict[key] = beta1 * proxy_dict[key] + (1 - beta1) * delta if round_idx > 0 else delta
        delta_square = torch.square(proxy_dict[key])
        opt_proxy_dict[key] = param - (1-beta2)*delta_square*torch.sign(param - delta_square)
        global_model_state_dict[key] += epsilon * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+tau)
    
    return global_model_state_dict


# def FedAdam(client_state_dicts: List[Dict], global_model_state_dict: Dict,
#                        server_velocity: Dict = None, server_momentum: Dict = None,
#                        learning_rate: float = 0.1, beta1: float = 0.9, beta2: float = 0.99,
#                        epsilon: float = 1e-8, tau: float = 0.0) -> Tuple[Dict, Dict, Dict]:
#     """
#     执行FedAdam策略的模型参数聚合 - Adam优化器的联邦版本
    
#     Args:
#         client_state_dicts: 客户端模型状态字典列表
#         global_model_state_dict: 全局模型状态字典
#         server_velocity: 服务器累积二阶动量
#         server_momentum: 服务器一阶动量
#         learning_rate: 学习率
#         beta1: 一阶动量衰减率
#         beta2: 二阶动量衰减率
#         epsilon: 数值稳定性参数
#         tau: 初始累积项
        
#     Returns:
#         (全局聚合后的状态字典, 更新后的服务器二阶动量, 更新后的服务器一阶动量)
#     """
#     # 1. 计算当前轮次的模型参数平均值
#     avg_state_dict = FedAvg(client_state_dicts, global_model_state_dict)
    
#     # 2. 初始化动量和速度
#     if server_momentum is None:
#         server_momentum = {}
#         for key, param in global_model_state_dict.items():
#             if isinstance(param, torch.Tensor):
#                 server_momentum[key] = torch.zeros_like(param)
#             else:
#                 server_momentum[key] = param
    
#     if server_velocity is None:
#         server_velocity = {}
#         for key, param in global_model_state_dict.items():
#             if isinstance(param, torch.Tensor):
#                 server_velocity[key] = tau * torch.ones_like(param)
#             else:
#                 server_velocity[key] = param
    
#     # 3. 应用Adam更新
#     new_global_state_dict = {}
#     new_server_momentum = {}
#     new_server_velocity = {}
    
#     for key in global_model_state_dict.keys():
#         if isinstance(global_model_state_dict[key], torch.Tensor):
#             # 计算参数更新值
#             delta = avg_state_dict[key] - global_model_state_dict[key]
            
#             # 更新一阶动量
#             new_server_momentum[key] = beta1 * server_momentum[key] + (1 - beta1) * delta
            
#             # 更新二阶动量 (Adam方式)
#             new_server_velocity[key] = beta2 * server_velocity[key] + (1 - beta2) * delta.pow(2)
            
#             # 应用Adam更新到全局模型
#             adaptive_lr = learning_rate / (torch.sqrt(new_server_velocity[key]) + epsilon)
#             new_global_state_dict[key] = global_model_state_dict[key] + adaptive_lr * new_server_momentum[key]
#         else:
#             new_global_state_dict[key] = avg_state_dict[key]
#             new_server_momentum[key] = server_momentum[key]
#             new_server_velocity[key] = server_velocity[key]
    
#     return new_global_state_dict, new_server_velocity, new_server_momentum


def FedAdam(client_state_dicts: List[Dict], global_model_state_dict: Dict, round_idx: int = 0, beta1: float = 0.9, beta2: float = 0.99,
                        epsilon: float = 1e-3, tau: float = 1e-3) -> Tuple[Dict, Dict, Dict]:
    opt_proxy_dict = None
    proxy_dict = None
    proxy_dict, opt_proxy_dict = {}, {}
    num_clients = len(client_state_dicts)
    for key in global_model_state_dict.keys():
        proxy_dict[key] = torch.zeros_like(global_model_state_dict[key])
        opt_proxy_dict[key] = torch.ones_like(global_model_state_dict[key]) * tau**2
    
    for key, param in opt_proxy_dict.items():
        delta = sum([(client_state_dicts[c][key] - global_model_state_dict[key]) 
                         for c in range(num_clients)]) / num_clients
        proxy_dict[key] = beta1 * proxy_dict[key] + (1 - beta1) * delta if round_idx > 0 else delta
        opt_proxy_dict[key] = beta2*param + (1-beta2)*torch.square(proxy_dict[key])
        global_model_state_dict[key] += epsilon * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+tau)
    
    return global_model_state_dict