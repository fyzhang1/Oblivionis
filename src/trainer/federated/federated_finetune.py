import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional

from trainer.base import FinetuneTrainer
from torch.utils.data import Dataset
from trainer.federated.federated_utils import *
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
import os

logger = logging.getLogger(__name__)

class FederatedFinetuneTrainer(FinetuneTrainer):
    """联邦学习微调训练器, 继承自FinetuneTrainer"""
    
    def __init__(
        self,
        model,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        args=None,
        evaluator=None,
        template_args=None,
        num_clients=3,
        aggregation_strategy="FedAvg",
        global_rounds=1,
        **kwargs
    ):
        """初始化联邦学习微调训练器"""
        self.federated_dataset = train_dataset if isinstance(train_dataset, dict) else None
        dummy_train_dataset = None
        if self.federated_dataset and 0 in self.federated_dataset:
            dummy_train_dataset = self.federated_dataset[0]
        
        super().__init__(
            model=model,
            train_dataset=dummy_train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=args,
            evaluator=evaluator,
            template_args=template_args,
        )
        
        self.num_clients = num_clients
        self.aggregation_strategy = aggregation_strategy
        self.global_rounds = global_rounds
        self.client_models = []
        self.client_data_sizes = []  # 记录每个客户端的样本量
        self.kwargs = kwargs
        
        # 检测是否为PEFT模型
        self.is_peft = isinstance(self.model, PeftModel)

        # self.server_momentum = None  # 用于FedAvgM、FedAdam、FedYogi
        # self.server_velocity = None  # 用于FedAdagrad、FedAdam、FedYogi
        
        # FedAdam专用的动量状态
        self.fedadam_momentum = None   # FedAdam的一阶动量
        self.fedadam_velocity = None   # FedAdam的二阶动量
        
        # FedYogi专用的动量状态
        self.fedyogi_momentum = None   # FedYogi的一阶动量
        self.fedyogi_velocity = None   # FedYogi的二阶动量
        
        # FedAvgM专用的动量状态
        self.fedavgm_momentum = None   # FedAvgM的动量状态
        
        # FedAdagrad专用的累积状态
        self.fedadagrad_velocity = None  # FedAdagrad的累积梯度平方和
        
        # 为算法特定参数设置默认值
        self.federated_args = {
            'server_lr': kwargs.get('server_lr', 1.0),
            'beta1': kwargs.get('beta1', 0.9),
            'beta2': kwargs.get('beta2', 0.99),
            'epsilon': kwargs.get('epsilon', 1e-3),
            'tau': kwargs.get('tau', 1e-3),
            'mu': kwargs.get('mu', 0.01),
            'momentum_factor': kwargs.get('momentum_factor', 0.9)
        }
        
        if not self.federated_dataset:
            logger.warning("No federated dataset provided!")
        else:
            for client_idx in range(num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Client {client_idx} has no data!")
        
        logger.info(f"Initialized with {num_clients} clients, global rounds: {global_rounds}")
        logger.info(f"Model type - PEFT: {self.is_peft}")

    def train(self, **kwargs):
        """执行联邦学习训练过程"""
                
        # 将全局模型移到CPU，以节省GPU内存
        self.model = self.model.cpu()
        self.model = deepcopy(self.model)
        
        for round_idx in range(self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
            # 在CPU上初始化客户端模型
            self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]
            self.client_data_sizes = []  # 重置每轮的样本量记录

            # 训练客户端模型
            logger.info(f"Aggregation strategy: {self.aggregation_strategy}")

            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Skipping client {client_idx} as no data is available")
                    self.client_data_sizes.append(0)  # 记录样本量为0
                    continue
                    
                client_model = self.client_models[client_idx]
                client_dataset = self.federated_dataset[client_idx]
                
                # 记录客户端样本量
                client_data_size = len(client_dataset)
                self.client_data_sizes.append(client_data_size)
                
                logger.info(f"Training on client {client_idx}")
                logger.info(f"Client {client_idx} dataset size: {client_data_size} samples")
                logger.info(f"Training args - Batch size: {self.args.per_device_train_batch_size}, Epochs: {self.args.num_train_epochs}")
                if hasattr(self.args, 'max_steps') and self.args.max_steps > 0:
                    logger.info(f"Max steps limit: {self.args.max_steps}")
                
                trained_client_model = self.train_client_model(client_model, client_dataset)
                
                # 训练完成后将训练后的模型保存到client_models
                self.client_models[client_idx] = trained_client_model
                # # 记录参与训练的客户端索引
                # self.trained_client_indices.append(client_idx)
            
            # 聚合模型 - 结果已经直接更新到self.model
            self.aggregate_models(round_idx)
            logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")
            
            # 保存状态
            if self.args.save_strategy == "epoch" or (round_idx == self.global_rounds - 1):
                save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
                # 确保模型在CPU上保存
                self.model = self.model.cpu()
                self.save_model(save_path)
                logger.info(f"Model saved at {save_path}")
        
        self.save_state()
        return None

    def train_client_model(self, client_model, client_dataset):
        """训练单个客户端模型"""
        if self.aggregation_strategy == "FedProx":
            
            class FedProxTrainer(FinetuneTrainer):
                def __init__(self, *args, global_model=None, mu=0.01, **kwargs):
                    super(FedProxTrainer, self).__init__(*args, **kwargs)
                    self.global_model = global_model
                    self.mu = mu
                    
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    return_values = super().compute_loss(model, inputs, return_outputs=return_outputs)
                    
                    if return_outputs:
                        loss, outputs = return_values
                    else:
                        loss = return_values

                    proximal_term = 0.0
                    if self.global_model is not None:
                        global_state = get_peft_model_state_dict(self.global_model)
                        for name, w in model.named_parameters():
                            if not w.requires_grad:
                                continue
                            name = name.replace(".default", "")  # 适配 PEFT
                            if name not in global_state:
                                logger.warning(f"Parameter {name} not found in global_state, skipping")
                                continue
                            # 只累加proximal term，不要在循环内修改loss
                            proximal_term += torch.norm(w - global_state[name].to(w.device).detach()) ** 2
                        
                        # 在循环外统一添加proximal term到loss
                        loss += (self.mu / 2) * proximal_term
                        # logger.info(f"FedProx loss: {loss.item()}, proximal_term: {proximal_term.item()}")
                    
                    return (loss, outputs) if return_outputs else loss
                
            trainer = FedProxTrainer(
                model=client_model,
                train_dataset=client_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
                global_model=self.model.cpu(),  # 传递CPU上的全局模型
                mu=self.federated_args['mu']
            )
        else:
            trainer = FinetuneTrainer(
                model=client_model,
                train_dataset=client_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
            )
        
        trainer.train()
        # return trainer
        
        # 重要：确保训练后的模型状态被正确返回
        trained_model = trainer.model.cpu()
        return trained_model

    def aggregate_models(self, round_idx=0):
        """聚合所有客户端模型"""
        logger.info("Aggregating client models")
        
        # 计算基于样本量的权重
        total_samples = sum(self.client_data_sizes)
        if total_samples == 0:
            logger.warning("No samples available for aggregation")
            return
            
        client_weights = [size / total_samples for size in self.client_data_sizes]
        logger.info(f"Client sample sizes: {self.client_data_sizes}")
        logger.info(f"Client weights: {[f'{w:.4f}' for w in client_weights]}")
        
        if self.is_peft:
            # 对于PEFT模型，只聚合可训练的参数（LoRA权重）
            client_state_dicts = []
            for model in self.client_models:
                model = model.cpu()
                # 只获取PEFT适配器的状态字典
                peft_state_dict = get_peft_model_state_dict(model)
                # 可以尝试下面
                # peft_state_dict = copy.deepcopy(get_peft_model_state_dict(model)
                
                if not peft_state_dict:
                    logger.warning(f"Client {model} has no PEFT state dict")
                    continue
            
                client_state_dicts.append(peft_state_dict)
                logger.debug(f"Client {model} has PEFT state dict")
        else:
            # 对于普通模型，聚合所有参数
            client_state_dicts = [model.cpu().state_dict() for model in self.client_models]
        
        logger.info(f"Aggregation strategy: {self.aggregation_strategy}")
        
        if self.is_peft:
            # 获取当前全局模型的PEFT状态字典
            global_peft_state_dict = get_peft_model_state_dict(self.model.cpu())
        else:
            global_peft_state_dict = self.model.cpu().state_dict()
        
        if self.aggregation_strategy == "FedAvg":
            global_state_dict = FedAvg(
                client_state_dicts, 
                global_model_state_dict=global_peft_state_dict,
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedAvgM":
            global_state_dict, self.fedavgm_momentum = FedAvgM(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                proxy_dict=self.fedavgm_momentum,
                round_idx=round_idx,
                momentum_factor=self.federated_args['momentum_factor'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedAdagrad":
            global_state_dict, self.fedadagrad_velocity = FedAdagrad(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                server_velocity=self.fedadagrad_velocity,
                epsilon=self.federated_args['epsilon'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedYogi":
            global_state_dict, self.fedyogi_momentum, self.fedyogi_velocity = FedYogi(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                proxy_dict=self.fedyogi_momentum,
                opt_proxy_dict=self.fedyogi_velocity,
                round_idx=round_idx,
                beta1=self.federated_args['beta1'],
                beta2=self.federated_args['beta2'],
                epsilon=self.federated_args['epsilon'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedAdam":
            global_state_dict, self.fedadam_momentum, self.fedadam_velocity = FedAdam(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                proxy_dict=self.fedadam_momentum,
                opt_proxy_dict=self.fedadam_velocity,
                round_idx=round_idx,
                beta1=self.federated_args['beta1'],
                beta2=self.federated_args['beta2'],
                epsilon=self.federated_args['epsilon'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )

        elif self.aggregation_strategy == "FedProx":
            # FedProx的聚合与FedAvg相同，使用样本量权重
            global_state_dict = FedAvg(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                client_weights=client_weights
            )
        
        if self.is_peft:
            # 对于PEFT模型，只加载PEFT适配器的权重
            # self.model.load_state_dict(global_state_dict, strict=False)
            set_peft_model_state_dict(self.model, global_state_dict)
        else:
            # 对于普通模型，加载所有权重
            self.model.load_state_dict(global_state_dict)
            
        logger.info("Global model updated")


    def save_model(self, output_dir=None):
        """保存模型，支持PEFT模型"""
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        
        if self.is_peft:
            # 保存PEFT模型
            logger.info(f"Saving PEFT model to {output_dir}")
            self.model.save_pretrained(output_dir)
            # 同时保存tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
        else:
            # 保存普通模型
            super().save_model(output_dir)