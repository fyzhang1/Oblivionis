import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional

from trainer.base import FinetuneTrainer
from trainer.unlearn.base import UnlearnTrainer
from data.unlearn import ForgetRetainDataset
from torch.utils.data import Dataset
from trainer.federated.federated_utils import *

logger = logging.getLogger(__name__)

class FederatedUnlearningTrainer(FinetuneTrainer):
    """联邦学习与遗忘训练器, 继承自FinetuneTrainer"""
    
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
            # dummy_train_dataset = self.federated_dataset[0].get("retain")
        
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
        
        # logger.info(f"Training args: {self.args}")
        self.num_clients = num_clients
        self.target_client_idx = target_client_idx
        self.unlearn_trainer_cls = unlearn_trainer_cls
        self.aggregation_strategy = aggregation_strategy
        self.global_rounds = global_rounds
        self.client_models = []
        self.kwargs = kwargs

        self.server_momentum = None  # 用于FedAvgM、FedAdam、FedYogi
        self.server_velocity = None  # 用于FedAdagrad、FedAdam、FedYogi
        self.server_control = None   # 用于SCAFFOLD
        self.client_controls = None  # 用于SCAFFOLD
        
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
        
        if not self.federated_dataset:
            logger.warning("No federated dataset provided!")
        else:
            for client_idx in range(num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Client {client_idx} has no data!")
        
        logger.info(f"Initialized with {num_clients} clients, target: {target_client_idx}, global rounds: {global_rounds}")
    
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        """执行联邦学习训练过程"""
        if not self.federated_dataset:
            raise ValueError("Federated dataset is not properly set up")
        
        logger.info(f"Starting federated unlearning training with {self.global_rounds} global rounds")
        
        # 初始化全局模型为我们当前模型
        self.model = deepcopy(self.model)
        
        for round_idx in range(self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
            # 每轮使用当前全局模型来初始化客户端模型
            self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]

            # 训练客户端模型
            logger.info(f"Aggregation_strategy: {self.aggregation_strategy}")

            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Skipping client {client_idx} as no data is available")
                    continue
                    
                client_model = self.client_models[client_idx]
                client_dataset = self.federated_dataset[client_idx]
                
                # federated learning
                if client_idx == self.target_client_idx:
                    logger.info(f"Unlearning on client {client_idx}")
                    self._unlearn_client_model(client_model, client_dataset)
                else:
                    logger.info(f"Training on client {client_idx}")
                    self._train_client_model(client_model, client_dataset)
            
            # 聚合模型 - 结果已经直接更新到self.model
            self._aggregate_models()
            logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")
            
            # 每轮保存一次状态
            if self.args.save_strategy == "epoch" or (round_idx == self.global_rounds - 1):
                save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
                self.save_model(save_path)
                logger.info(f"Model saved at {save_path}")
        
        self.save_state()
        return None

    def _unlearn_client_model(self, client_model, client_dataset):
        from trainer import TRAINER_REGISTRY
        trainer_cls = TRAINER_REGISTRY.get(self.unlearn_trainer_cls, None)
        

        # <class 'data.unlearn.ForgetRetainDataset'>
        forget_dataset = client_dataset.forget
        retain_dataset = client_dataset.retain
        
        logger.info(f"forget_data length: {len(forget_dataset)}")
        logger.info(f"retain_data length: {len(retain_dataset)}")
        # logger.info(f"forget_data type: {type(forget_dataset)}") <class 'data.unlearn.ForgetRetainDataset'>
        # logger.info(f"retain_data type: {type(retain_dataset)}") <class 'data.unlearn.ForgetRetainDataset'>
        # logger.info(f"client_dataset type: {type(client_dataset)}") <class 'data.unlearn.ForgetRetainDataset'>


        # forget_data = forget_dataset.forget
        # retain_data = retain_dataset.retain
        # print(forget_data[0])
        # print(retain_data[0])

        class CombinedDataset(Dataset):
            """合并 Forget 和 Retain 数据集的适配器"""
            def __init__(self, forget_data, retain_data):
                self.forget_data = forget_data
                self.retain_data = retain_data
                # assert len(forget_data) == len(retain_data), "数据长度必须一致"

            def __len__(self):
                return len(self.forget_data)

            def __getitem__(self, idx):
                return {
                    "forget": self.forget_data[idx],
                    "retain": self.retain_data[idx]
                }
        

        # 创建组合数据集
        combined_dataset = CombinedDataset(forget_dataset, retain_dataset)
        print(type(combined_dataset))
        print(self.unlearn_trainer_cls)

        logger.info(f"Unlearning_cls_name: {self.unlearn_trainer_cls}")
        
        # 创建训练器
        unlearn_trainer = trainer_cls(
            model=client_model,
            train_dataset=combined_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=self.args,
            evaluator=self.evaluator,
            template_args=self.template_args,
            **self.kwargs
        )
        
        # 执行训练
        unlearn_trainer.train()
        
        return unlearn_trainer
    

    def _train_client_model(self, client_model, client_dataset):
        retain_dataset = client_dataset.retain
        # if retain_data is None:
        #     raise ValueError(f"Client has no retain data!")

        retain_data = retain_dataset.retain
        
        # 提取 'retain' 子字典
        # logger.info(f"retain_data type: {type(retain_data)}")
        # logger.info(f"retain_data length: {len(retain_data)}")
        # sample = retain_data[0]
        # logger.info(f"Sample type: {type(sample)}")
        # logger.info(f"Sample content: {sample}")
        
        # 转换为干净的样本列表
        # filtered_retain_data = [sample['retain'] for sample in retain_data]

        # logger.info(f"Filtered retain_data sample: {filtered_retain_data[0]}")
        
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
        # for key, param in client_model.state_dict().items():
        #     if param.numel() == 0:
        #         logger.warning(f"Client model has empty parameter after training: {key}")
        return trainer
    

    def _aggregate_models(self):

        # fedavg
        logger.info("Aggregating client models")
        client_state_dicts = [model.state_dict() for model in self.client_models]
        logger.info(f"Aggregation_strategy:{self.aggregation_strategy }")
        
        if self.aggregation_strategy == "FedAvg":
            global_state_dict = FedAvg(
            client_state_dicts, 
            global_model_state_dict=self.model.state_dict()
            )
        elif self.aggregation_strategy == "FedAvgM":
            global_state_dict, self.server_momentum = FedAvgM(
                    client_state_dicts,
                    global_model_state_dict=self.model.state_dict(),
                    server_momentum=self.server_momentum,
                    momentum_factor=self.fed_args['momentum_factor']
                )
                
        elif self.aggregation_strategy == "FedAdagrad":
                        global_state_dict, self.server_velocity = FedAdagrad(
                            client_state_dicts,
                            global_model_state_dict=self.model.state_dict(),
                            server_velocity=self.server_velocity,
                            learning_rate=self.fed_args['server_lr'],
                            epsilon=self.fed_args['epsilon'],
                            tau=self.fed_args['tau']
                )
                
        elif self.aggregation_strategy == "FedYogi":
                global_state_dict, self.server_velocity, self.server_momentum = FedYogi(
                    client_state_dicts,
                    global_model_state_dict=self.model.state_dict(),
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
                    global_model_state_dict=self.model.state_dict(),
                    server_velocity=self.server_velocity,
                    server_momentum=self.server_momentum,
                    learning_rate=self.fed_args['server_lr'],
                    beta1=self.fed_args['beta1'],
                    beta2=self.fed_args['beta2'],
                    epsilon=self.fed_args['epsilon'],
                    tau=self.fed_args['tau']
                )
        
        self.model.load_state_dict(global_state_dict)
        logger.info("Global model updated")