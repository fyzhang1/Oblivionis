import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, Trainer, TrainingArguments

from src.trainers.base_trainer import FinetuneTrainer
from src.utils.federated_utils import (
    get_federated_data, 
    fedavg_aggregate, 
    get_model_difference, 
    apply_model_difference,
    apply_differential_privacy
)

logger = logging.getLogger(__name__)


class FederatedUnlearningTrainer:
    """
    联邦未学习训练器，用于协调多个客户端的训练和未学习过程
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        num_clients: int = 5,
        target_client_idx: int = 0,
        client_epochs: int = 1,
        aggregation_strategy: str = "fedavg",
        privacy_enabled: bool = False,
        noise_multiplier: float = 0.1,
        max_grad_norm: float = 1.0,
        **kwargs
    ):
        """
        初始化联邦未学习训练器
        
        Args:
            model: 预训练模型
            args: 训练参数
            num_clients: 客户端数量
            target_client_idx: 目标客户端索引（包含遗忘数据的客户端）
            client_epochs: 每个客户端在本地训练的轮数
            aggregation_strategy: 模型聚合策略，默认为fedavg
            privacy_enabled: 是否启用差分隐私
            noise_multiplier: 差分隐私的噪声乘数
            max_grad_norm: 梯度裁剪的最大范数
            **kwargs: 其他参数传递给基础训练器
        """
        self.model = model
        self.args = args
        self.num_clients = num_clients
        self.target_client_idx = target_client_idx
        self.client_epochs = client_epochs
        self.aggregation_strategy = aggregation_strategy
        self.privacy_enabled = privacy_enabled
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.kwargs = kwargs
        
        # 保存原始的全局模型
        self.global_model = model
        
        # 为每个客户端创建单独的训练器
        self.client_trainers = {}
        
        # 保存目标客户端的训练前状态，用于检测模型变化
        self.target_client_initial_state = None
        
        logger.info(f"初始化联邦未学习训练器，客户端数量: {num_clients}，目标客户端索引: {target_client_idx}")
    
    def prepare_federated_data(
        self, 
        train_dataset: Union[Dataset, Dict[str, Dataset]], 
        eval_dataset: Optional[Dataset] = None,
        forget_dataset: Optional[Dataset] = None
    ):
        """
        准备联邦数据分割
        
        Args:
            train_dataset: 训练数据集或包含'retain'和'forget'键的字典
            eval_dataset: 评估数据集
            forget_dataset: 遗忘数据集（当train_dataset不是字典时使用）
        """
        logger.info("准备联邦数据分割...")
        
        # 使用联邦工具函数分割数据
        self.federated_train_data = get_federated_data(
            dataset=train_dataset,
            num_clients=self.num_clients,
            target_client_idx=self.target_client_idx,
            forget_dataset=forget_dataset
        )
        
        # 评估数据集由所有客户端共享
        self.eval_dataset = eval_dataset
        
        # 为每个客户端创建训练器
        logger.info("为每个客户端创建训练器...")
        for client_idx in range(self.num_clients):
            client_args = self.args
            
            # 创建客户端训练器
            if client_idx == self.target_client_idx:
                # 目标客户端使用未学习训练器
                self.client_trainers[client_idx] = FinetuneTrainer(
                    model=self._get_client_model(client_idx),
                    args=client_args,
                    train_dataset=self.federated_train_data[client_idx],
                    eval_dataset=self.eval_dataset,
                    **self.kwargs
                )
                logger.info(f"创建目标客户端(#{client_idx})训练器，数据集大小: {len(self.federated_train_data[client_idx])}")
            else:
                # 非目标客户端使用普通训练器
                self.client_trainers[client_idx] = FinetuneTrainer(
                    model=self._get_client_model(client_idx),
                    args=client_args,
                    train_dataset=self.federated_train_data[client_idx],
                    eval_dataset=self.eval_dataset,
                    **self.kwargs
                )
                logger.info(f"创建客户端(#{client_idx})训练器，数据集大小: {len(self.federated_train_data[client_idx])}")
    
    def _get_client_model(self, client_idx: int) -> PreTrainedModel:
        """
        获取客户端模型（深度复制全局模型）
        
        Args:
            client_idx: 客户端索引
            
        Returns:
            客户端模型副本
        """
        return self._copy_model(self.global_model)
    
    def _copy_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        深度复制模型
        
        Args:
            model: 要复制的模型
            
        Returns:
            复制的模型
        """
        model_copy = type(model)(model.config)
        model_copy.load_state_dict(model.state_dict())
        return model_copy
    
    def train(self, **kwargs):
        """
        执行联邦训练过程
        
        Returns:
            训练结果
        """
        logger.info("开始联邦训练...")
        
        # 记录训练指标
        metrics = {}
        
        # 执行联邦训练循环
        for epoch in range(self.client_epochs):
            logger.info(f"联邦训练轮次 {epoch+1}/{self.client_epochs}")
            
            # 训练每个客户端
            client_models = []
            client_weights = []
            
            for client_idx, client_trainer in self.client_trainers.items():
                logger.info(f"训练客户端 #{client_idx}...")
                
                # 如果是第一轮且是目标客户端，保存初始状态
                if epoch == 0 and client_idx == self.target_client_idx:
                    self.target_client_initial_state = self._copy_model(client_trainer.model)
                
                # 使用客户端本地数据训练
                train_results = client_trainer.train(**kwargs)
                
                # 应用差分隐私（如果启用）
                if self.privacy_enabled and client_idx != self.target_client_idx:
                    client_trainer.model = apply_differential_privacy(
                        client_trainer.model,
                        self.noise_multiplier,
                        self.max_grad_norm
                    )
                    logger.info(f"对客户端 #{client_idx} 应用差分隐私")
                
                # 收集客户端模型和权重
                client_models.append(client_trainer.model)
                client_weights.append(len(self.federated_train_data[client_idx]))
                
                # 记录客户端训练指标
                for key, value in train_results.metrics.items():
                    metrics[f"client_{client_idx}_{key}"] = value
            
            # 使用指定策略聚合模型
            if self.aggregation_strategy == "fedavg":
                # 计算标准化权重
                normalized_weights = [w / sum(client_weights) for w in client_weights]
                
                # 聚合客户端模型
                self.global_model = fedavg_aggregate(
                    client_models=client_models,
                    global_model=self.global_model,
                    weights=normalized_weights
                )
                logger.info("使用联邦平均算法聚合客户端模型")
            else:
                raise ValueError(f"不支持的聚合策略: {self.aggregation_strategy}")
            
            # 将更新后的全局模型分发给各客户端
            for client_idx, client_trainer in self.client_trainers.items():
                client_trainer.model.load_state_dict(self.global_model.state_dict())
            
            # 评估全局模型
            if self.eval_dataset is not None:
                logger.info("评估全局模型...")
                eval_trainer = FinetuneTrainer(
                    model=self.global_model,
                    args=self.args,
                    eval_dataset=self.eval_dataset,
                    **self.kwargs
                )
                eval_results = eval_trainer.evaluate()
                
                # 记录全局模型评估指标
                for key, value in eval_results.items():
                    metrics[f"global_{key}_epoch_{epoch}"] = value
        
        return metrics
    
    def unlearn(self, **kwargs):
        """
        执行联邦未学习过程
        
        Returns:
            未学习结果
        """
        logger.info("开始联邦未学习...")
        
        # 确保目标客户端已经完成训练
        if self.target_client_initial_state is None:
            raise ValueError("必须先调用train()方法进行训练，然后才能执行未学习")
        
        # 获取目标客户端的模型
        target_client_trainer = self.client_trainers[self.target_client_idx]
        target_client_model = target_client_trainer.model
        
        # 计算目标客户端在训练前后的模型差异
        model_diff = get_model_difference(target_client_model, self.target_client_initial_state)
        
        # 从全局模型中移除目标客户端的贡献
        logger.info("从全局模型中移除目标客户端的贡献...")
        client_weight = len(self.federated_train_data[self.target_client_idx])
        total_weight = sum([len(data) for data in self.federated_train_data.values()])
        
        # 计算要移除的贡献比例
        contribution_scale = -1.0 * (client_weight / total_weight)
        
        # 应用反向差异到全局模型
        self.global_model = apply_model_difference(
            self.global_model, 
            model_diff, 
            scale=contribution_scale
        )
        
        logger.info(f"已从全局模型中移除客户端 #{self.target_client_idx} 的贡献")
        
        # 重新分发更新后的全局模型到非目标客户端
        for client_idx, client_trainer in self.client_trainers.items():
            if client_idx != self.target_client_idx:
                client_trainer.model.load_state_dict(self.global_model.state_dict())
        
        # 评估未学习后的全局模型
        metrics = {}
        if self.eval_dataset is not None:
            logger.info("评估未学习后的全局模型...")
            eval_trainer = FinetuneTrainer(
                model=self.global_model,
                args=self.args,
                eval_dataset=self.eval_dataset,
                **self.kwargs
            )
            eval_results = eval_trainer.evaluate()
            
            # 记录全局模型评估指标
            for key, value in eval_results.items():
                metrics[f"unlearned_global_{key}"] = value
        
        return metrics
    
    def save_model(self, output_dir: str):
        """
        保存全局模型
        
        Args:
            output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存全局模型
        global_model_path = os.path.join(output_dir, "global_model")
        self.global_model.save_pretrained(global_model_path)
        logger.info(f"全局模型已保存到 {global_model_path}")
        
        # 保存每个客户端的模型
        for client_idx, client_trainer in self.client_trainers.items():
            client_model_path = os.path.join(output_dir, f"client_{client_idx}_model")
            client_trainer.model.save_pretrained(client_model_path)
            logger.info(f"客户端 #{client_idx} 模型已保存到 {client_model_path}")
    
    def evaluate(self, **kwargs):
        """
        评估全局模型
        
        Returns:
            评估结果
        """
        if self.eval_dataset is None:
            logger.warning("未提供评估数据集，跳过评估")
            return {}
        
        logger.info("评估全局模型...")
        eval_trainer = FinetuneTrainer(
            model=self.global_model,
            args=self.args,
            eval_dataset=self.eval_dataset,
            **self.kwargs
        )
        return eval_trainer.evaluate(**kwargs) 