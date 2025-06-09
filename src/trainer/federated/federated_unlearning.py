import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional

from trainer.base import BaseTrainer, FinetuneTrainer
from trainer.unlearn.base import UnlearnTrainer
from data.unlearn import ForgetRetainDataset
from torch.utils.data import Dataset
from trainer.federated.federated_utils import *
from transformers import TrainingArguments
from vllm import SamplingParams

logger = logging.getLogger(__name__)

class FederatedUnlearningTrainer(BaseTrainer):
    """联邦学习与遗忘训练器"""
    
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
        target_client_idx=0,
        unlearn_trainer_cls=None,
        aggregation_strategy="average",
        global_rounds=1,
        **kwargs
    ):
        """初始化联邦学习与遗忘训练器"""
        self.federated_dataset = train_dataset if isinstance(train_dataset, dict) else None
        dummy_train_dataset = None
        if self.federated_dataset and 0 in self.federated_dataset:
            dummy_train_dataset = self.federated_dataset[0].retain
        
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
        self.target_client_idx = target_client_idx
        self.unlearn_trainer_cls = unlearn_trainer_cls
        self.aggregation_strategy = aggregation_strategy
        self.global_rounds = global_rounds
        self.client_models = []
        self.kwargs = kwargs
        
        self.is_vllm = not hasattr(self.model, 'forward')
        
        if not self.federated_dataset:
            logger.warning("No federated dataset provided!")
        else:
            for client_idx in range(num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Client {client_idx} has no data!")
        
        logger.info(f"Initialized with {num_clients} clients, target: {target_client_idx}, global rounds: {global_rounds}")
        
        self.server_momentum = None  # 用于FedAvgM、FedAdam、FedYogi
        self.server_velocity = None  # 用于FedAdagrad、FedAdam、FedYogi
        # self.server_control = None   # 用于SCAFFOLD
        # self.client_controls = None  # 用于SCAFFOLD
        
        # 为算法特定参数设置默认值
        self.fed_args = {
            'server_lr': kwargs.get('server_lr', 1.0),
            'beta1': kwargs.get('beta1', 0.9),
            'beta2': kwargs.get('beta2', 0.99),
            'epsilon': kwargs.get('epsilon', 1e-8),
            'tau': kwargs.get('tau', 0.0),
            'mu': kwargs.get('mu', 0.01),
            'momentum_factor': kwargs.get('momentum_factor', 0.9)
        }

    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        """执行联邦学习训练过程"""
        if not self.federated_dataset:
            raise ValueError("Federated dataset is not properly set up")
        
        logger.info(f"Starting federated unlearning training with {self.global_rounds} global rounds")
        
        # 将全局模型移到CPU，以节省GPU内存
        if not self.is_vllm:
            self.model = self.model.cpu()
            self.model = deepcopy(self.model)

        # 步骤1: 目标客户端执行unlearning
            logger.info(f"Step 1: Unlearning on target client {self.target_client_idx}")
            if self.target_client_idx in self.federated_dataset:
                target_dataset = self.federated_dataset[self.target_client_idx]
                # 使用当前全局模型进行unlearning
                self._unlearn_client_model(deepcopy(self.model), target_dataset)
                
                # 步骤2: 将unlearning后的模型设为新的全局模型
                # self.model = self.model
                logger.info("Updated global model with unlearned model")
            else:
                logger.warning(f"Target client {self.target_client_idx} has no data, skipping unlearning")
        
        for round_idx in range(self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
            # 步骤3: 所有客户端基于新全局模型在retain data上训练
            logger.info("Step 2: All clients training on retain data with updated global model")
            
            # 在CPU上初始化客户端模型（基于unlearning后的全局模型）
            if not self.is_vllm:
                self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]
            else:
                self.client_models = [self.model for _ in range(self.num_clients)]

            # 所有客户端在retain data上训练
            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Skipping client {client_idx} as no data is available")
                    continue
                    
                client_model = self.client_models[client_idx]
                client_dataset = self.federated_dataset[client_idx]
                
                logger.info(f"Training client {client_idx} on retain data")
                self._train_client_model(client_model, client_dataset)
                
                # 训练完成后将模型移回CPU
                if not self.is_vllm:
                    self.client_models[client_idx] = client_model.cpu()
            
            # 步骤4: 聚合所有客户端模型
            logger.info("Step 3: Aggregating all client models")
            self._aggregate_models()
            logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")
            
            # 保存状态
            if self.args.save_strategy == "epoch" or (round_idx == self.global_rounds - 1):
                save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
                # 确保模型在CPU上保存
                if not self.is_vllm:
                    self.model = self.model.cpu()
                self.save_model(save_path)
                logger.info(f"Model saved at {save_path}")
        
        self.save_state()
        return None
    
    # def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
    #     """执行联邦学习训练过程"""
    #     if not self.federated_dataset:
    #         raise ValueError("Federated dataset is not properly set up")
        
    #     logger.info(f"Starting federated unlearning training with {self.global_rounds} global rounds")
        
    #     # 将全局模型移到CPU，以节省GPU内存
    #     if not self.is_vllm:
    #         self.model = self.model.cpu()
    #         self.model = deepcopy(self.model)
        
    #     for round_idx in range(self.global_rounds):
    #         logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
    #         # 在CPU上初始化客户端模型
    #         if not self.is_vllm:
    #             self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]
    #         else:
    #             # 对于vLLM模型，我们只保存状态字典
    #             self.client_models = [self.model for _ in range(self.num_clients)]

    #         # 训练客户端模型
    #         logger.info(f"Aggregation_strategy: {self.aggregation_strategy}")

    #         for client_idx in range(self.num_clients):
    #             if client_idx not in self.federated_dataset:
    #                 logger.warning(f"Skipping client {client_idx} as no data is available")
    #                 continue
                    
    #             client_model = self.client_models[client_idx]
    #             client_dataset = self.federated_dataset[client_idx]
                
    #             # federated learning
    #             if client_idx == self.target_client_idx:
    #                 logger.info(f"Unlearning on client {client_idx}")
    #                 self._unlearn_client_model(client_model, client_dataset)
    #             else:
    #                 logger.info(f"Training on client {client_idx}")
    #                 self._train_client_model(client_model, client_dataset)

                
    #             # 训练完成后将模型移回CPU
    #             if not self.is_vllm:
    #                 self.client_models[client_idx] = client_model.cpu()
            
    #         # 聚合模型 - 结果已经直接更新到self.model
    #         self._aggregate_models()
    #         logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")
            
    #         # 保存状态
    #         if self.args.save_strategy == "epoch" or (round_idx == self.global_rounds - 1):
    #             save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
    #             # 确保模型在CPU上保存
    #             if not self.is_vllm:
    #                 self.model = self.model.cpu()
    #             self.save_model(save_path)
    #             logger.info(f"Model saved at {save_path}")
        
    #     self.save_state()
    #     return None

    def _unlearn_client_model(self, client_model, client_dataset):
        from trainer import TRAINER_REGISTRY
        trainer_cls = TRAINER_REGISTRY.get(self.unlearn_trainer_cls, None)
        
        forget_dataset = client_dataset.forget
        retain_dataset = client_dataset.retain
        
        logger.info(f"forget_data length: {len(forget_dataset)}")
        logger.info(f"retain_data length: {len(retain_dataset)}")
        logger.info(f"Unlearning_cls_name: {self.unlearn_trainer_cls}")

        # 准备训练参数
        training_args = self.args
        if hasattr(training_args, 'deepspeed'):
            # 如果存在deepspeed配置，创建一个新的TrainingArguments实例
            training_args_dict = training_args.to_dict()
            if 'deepspeed' in training_args_dict:
                del training_args_dict['deepspeed']
            # 禁用梯度累积，因为DeepSpeed已经处理了这个问题
            training_args_dict['gradient_accumulation_steps'] = 1
            from transformers import TrainingArguments
            training_args = TrainingArguments(**training_args_dict)

        if self.aggregation_strategy == "FedProx":
            class FedProxUnlearnTrainer(trainer_cls):
                def __init__(self, *args, global_model=None, mu=0.01, **kwargs):
                    # 移除 deepspeed 相关配置
                    if 'deepspeed' in kwargs:
                        del kwargs['deepspeed']
                    super().__init__(*args, **kwargs)
                    self.global_model = global_model.to(self.args.device) if global_model is not None else None
                    self.mu = mu
                
                def compute_loss(self, model, inputs, return_outputs=False):
                    # 获取原始损失
                    loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
                    
                    # 添加FedProx正则项
                    proximal_term = 0.0
                    if self.global_model is not None:
                        for w, w_t in zip(model.parameters(), self.global_model.parameters()):
                            if w.requires_grad:
                                proximal_term += torch.norm(w - w_t.detach()) ** 2
                        
                        proximal_term = (self.mu / 2) * proximal_term
                        loss += proximal_term
                    
                    return (loss, outputs) if return_outputs else loss
            
            unlearn_trainer = FedProxUnlearnTrainer(
                model=client_model,
                train_dataset=forget_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=training_args,
                evaluator=self.evaluator,
                template_args=self.template_args,
                global_model=self.model.cpu(),  # 传递CPU上的全局模型
                mu=self.fed_args['mu'],
                **self.kwargs
            )
        else:
            # 移除 deepspeed 相关配置
            kwargs = self.kwargs.copy()
            if 'deepspeed' in kwargs:
                del kwargs['deepspeed']
                
            unlearn_trainer = trainer_cls(
                model=client_model,
                train_dataset=forget_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=training_args,
                evaluator=self.evaluator,
                template_args=self.template_args,
                **kwargs
            )
        
        # 训练器会自动将模型移到正确的设备上
        unlearn_trainer.train()
        
        return unlearn_trainer
    

    def _train_client_model(self, client_model, client_dataset):
        retain_dataset = client_dataset.retain
        retain_data = retain_dataset.retain
        
        if self.aggregation_strategy == "FedProx":
            class FedProxTrainer(FinetuneTrainer):
                def __init__(self, *args, global_model=None, mu=0.01, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.global_model = global_model.to(self.args.device) if global_model is not None else None
                    self.mu = mu
                
                def compute_loss(self, model, inputs, return_outputs=False):
                    outputs = model(**inputs)
                    loss = outputs.loss
                    
                    proximal_term = 0.0
                    if self.global_model is not None:
                        for w, w_t in zip(model.parameters(), self.global_model.parameters()):
                            if w.requires_grad:
                                proximal_term += torch.norm(w - w_t.detach()) ** 2
                                
                        proximal_term = (self.mu / 2) * proximal_term
                        loss += proximal_term
                        
                    return (loss, outputs) if return_outputs else loss
                
            trainer = FedProxTrainer(
                model=client_model,
                train_dataset=retain_data,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
                global_model=self.model.cpu(),  # 传递CPU上的全局模型
                mu=self.fed_args['mu']
            )
        else:
            trainer = FinetuneTrainer(
                model=client_model,
                train_dataset=retain_data,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
            )
        
        trainer.train()
        return trainer
    

    def _aggregate_models(self):
        logger.info("Aggregating client models")
        # 确保所有模型都在CPU上进行聚合
        client_state_dicts = [model.cpu().state_dict() for model in self.client_models]
        logger.info(f"Aggregation_strategy:{self.aggregation_strategy}")
        
        if self.aggregation_strategy == "FedAvg":
            global_state_dict = FedAvg(
                client_state_dicts, 
                global_model_state_dict=self.model.cpu().state_dict()
            )
        elif self.aggregation_strategy == "FedAvgM":
            global_state_dict, self.server_momentum = FedAvgM(
                client_state_dicts,
                global_model_state_dict=self.model.cpu().state_dict(),
                server_momentum=self.server_momentum,
                momentum_factor=self.fed_args['momentum_factor']
            )
        elif self.aggregation_strategy == "FedAdagrad":
            global_state_dict, self.server_velocity = FedAdagrad(
                client_state_dicts,
                global_model_state_dict=self.model.cpu().state_dict(),
                server_velocity=self.server_velocity,
                learning_rate=self.fed_args['server_lr'],
                epsilon=self.fed_args['epsilon'],
                tau=self.fed_args['tau']
            )
        elif self.aggregation_strategy == "FedYogi":
            global_state_dict, self.server_velocity, self.server_momentum = FedYogi(
                client_state_dicts,
                global_model_state_dict=self.model.cpu().state_dict(),
                server_velocity=self.server_velocity,
                server_momentum=self.server_momentum,
                learning_rate=self.fed_args['server_lr'],
                beta1=self.fed_args['beta1'],
                beta2=self.fed_args['beta2'],
                epsilon=self.fed_args['epsilon'],
                tau=self.fed_args['tau']
            )
        elif self.aggregation_strategy == "FedAdam":
            global_state_dict, self.server_velocity, self.server_momentum = FedAdam(
                client_state_dicts,
                global_model_state_dict=self.model.cpu().state_dict(),
                server_velocity=self.server_velocity,
                server_momentum=self.server_momentum,
                learning_rate=self.fed_args['server_lr'],
                beta1=self.fed_args['beta1'],
                beta2=self.fed_args['beta2'],
                epsilon=self.fed_args['epsilon'],
                tau=self.fed_args['tau']
            )
        elif self.aggregation_strategy == "FedProx":
            global_state_dict = FedProx(
                client_state_dicts,
                global_model_state_dict=self.model.cpu().state_dict(),
                mu=self.fed_args['mu']
            )
            
        self.model.load_state_dict(global_state_dict)
        logger.info("Global model updated")

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to handle vLLM models."""
        if not self.is_vllm:
            return super().compute_loss(model, inputs, return_outputs)
            
        # For vLLM models, we need to implement custom loss computation
        # This is a placeholder implementation - you'll need to modify this
        # based on your specific requirements
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            labels = inputs.get("labels")
            
            if input_ids is not None and labels is not None:
                # Create sampling parameters
                sampling_params = SamplingParams(
                    temperature=0.0,  # Use 0 temperature for deterministic output
                    max_tokens=labels.shape[1],
                    stop=None
                )
                
                # Generate outputs using vLLM
                outputs = model.generate(input_ids, sampling_params)
                
                # Compute loss (this is a placeholder - implement your actual loss computation)
                # You might need to convert the outputs to the format expected by your loss function
                loss = torch.tensor(0.0, device=input_ids.device)  # Placeholder
                
                if return_outputs:
                    return loss, outputs
                return loss
                
        raise NotImplementedError("Loss computation for vLLM models needs to be implemented for your specific use case")