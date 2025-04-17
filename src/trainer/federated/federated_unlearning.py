import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional

from trainer.base import FinetuneTrainer
from trainer.unlearn.base import UnlearnTrainer
from data.unlearn import ForgetRetainDataset
from torch.utils.data import Dataset
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
        self.client_models = []
        self.kwargs = kwargs
        
        if not self.federated_dataset:
            logger.warning("No federated dataset provided!")
        else:
            for client_idx in range(num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Client {client_idx} has no data!")
        
        logger.info(f"Initialized with {num_clients} clients, target: {target_client_idx}")
    
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        """执行联邦学习训练过程"""
        if not self.federated_dataset:
            raise ValueError("Federated dataset is not properly set up")
        
        logger.info("Starting federated unlearning training")
        self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]
        

        # 验证复制的模型
        # for idx, client_model in enumerate(self.client_models):
        #     for key, param in client_model.state_dict().items():
        #         if param.numel() == 0:
        #             logger.error(f"Client {idx} model has empty parameter after deepcopy: {key}")
        #         else:
        #             logger.info(f"Client {idx} parameter {key} size: {param.size()}")

        # 训练客户端模型
        for client_idx in range(self.num_clients):
            if client_idx not in self.federated_dataset:
                logger.warning(f"Skipping client {client_idx} as no data is available")
                continue
                
            client_model = self.client_models[client_idx]
            client_dataset = self.federated_dataset[client_idx]
            
            if client_idx == self.target_client_idx:
                logger.info(f"Unlearning on client {client_idx}")
                self._unlearn_client_model(client_model, client_dataset)
            else:
                logger.info(f"Training on client {client_idx}")
                self._train_client_model(client_model, client_dataset)
        
        # 聚合模型
        self._aggregate_models()
        self.save_state()
        return None

    def _unlearn_client_model(self, client_model, client_dataset):
        from trainer import TRAINER_REGISTRY
        trainer_cls = TRAINER_REGISTRY.get(self.unlearn_trainer_cls, None)
        
        # 获取 forget 和 retain 数据
        forget_dataset = client_dataset.forget
        retain_dataset = client_dataset.retain
        
        logger.info(f"forget_data length: {len(forget_dataset)}")
        logger.info(f"retain_data length: {len(retain_dataset)}")
        logger.info(f"forget_data type: {type(forget_dataset)}")
        logger.info(f"retain_data type: {type(retain_dataset)}")
        logger.info(f"client_dataset type: {type(client_dataset)}")
        # print(forget_dataset[0])

        forget_data = forget_dataset.forget
        retain_data = retain_dataset.retain
        print(forget_data[0])
        print(retain_data[0])

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
        if retain_data is None:
            raise ValueError(f"Client has no retain data!")

        retain_data = retain_dataset.retain
        
        # 提取 'retain' 子字典
        logger.info(f"retain_data type: {type(retain_data)}")
        logger.info(f"retain_data length: {len(retain_data)}")
        sample = retain_data[0]
        logger.info(f"Sample type: {type(sample)}")
        logger.info(f"Sample content: {sample}")
        
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
    
    # def _train_client_model(self, client_model, client_dataset):
    #     """对客户端执行正常训练，使用 retain 数据"""
    #     retain_data = client_dataset.get("retain")
    #     if retain_data is None:
    #         raise ValueError(f"Client has no retain data!")

    #     # 调试 retain_data
    #     logger.info(f"retain_data type: {type(retain_data)}")
    #     logger.info(f"retain_data length: {len(retain_data)}")
    #     sample = retain_data[0]
    #     logger.info(f"Sample type: {type(sample)}")
    #     logger.info(f"Sample content: {sample}")
        
    #     trainer = FinetuneTrainer(
    #         model=client_model,
    #         train_dataset=retain_data,
    #         tokenizer=self.tokenizer,
    #         data_collator=self.data_collator,
    #         args=self.args,
    #         evaluator=self.evaluator,
    #         template_args=self.template_args,
    #     )
    #     trainer.train()
    #     return trainer
    
    # def _aggregate_models(self):
    #     """聚合所有客户端模型，更新全局模型"""
    #     logger.info("Aggregating client models")
    #     client_state_dicts = [model.state_dict() for model in self.client_models]
        
    #     if self.aggregation_strategy == "average":
    #         global_state_dict = {}
    #         for key in client_state_dicts[0].keys():
    #             if isinstance(client_state_dicts[0][key], torch.Tensor):
    #                 global_state_dict[key] = torch.stack([
    #                     state_dict[key] for state_dict in client_state_dicts
    #                 ]).mean(dim=0)
    #             else:
    #                 global_state_dict[key] = client_state_dicts[0][key]
    #     else:
    #         raise NotImplementedError(f"Strategy {self.aggregation_strategy} not implemented")
        
    #     self.model.load_state_dict(global_state_dict)
    #     logger.info("Global model updated")

    def _aggregate_models(self):
        logger.info("Aggregating client models")
        client_state_dicts = [model.state_dict() for model in self.client_models]
        
        if self.aggregation_strategy == "average":
            global_state_dict = {}
            for key in client_state_dicts[0].keys():
                if isinstance(client_state_dicts[0][key], torch.Tensor):
                    valid_params = [state_dict[key] for state_dict in client_state_dicts if state_dict[key].numel() > 0]
                    if valid_params:
                        global_state_dict[key] = torch.stack(valid_params).mean(dim=0)
                    else:
                        global_state_dict[key] = self.model.state_dict()[key]
                        logger.warning(f"All clients have empty parameter for {key}, using global model parameter")
                else:
                    global_state_dict[key] = client_state_dicts[0][key]
        else:
            raise NotImplementedError(f"Strategy {self.aggregation_strategy} not implemented")
        
        self.model.load_state_dict(global_state_dict)
        logger.info("Global model updated")