import hydra
from omegaconf import DictConfig
from data import get_data, get_collators, get_federated_data
from model import get_model
from trainer import load_trainer
from evals import get_evaluator
from trainer.utils import seed_everything
from torch.utils.data import Subset
import numpy as np
import logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    
    # 添加调试信息
    logger.info(f"Model Configuration: {model_cfg}")
    logger.info(f"Model Args: {model_cfg.model_args}")
    logger.info(f"Use LoRA: {model_cfg.model_args.get('use_lora', 'NOT_FOUND')}")
    logger.info(f"LoRA Configuration: {model_cfg.model_args.get('lora_config', 'NOT_FOUND')}")
    
    model, tokenizer = get_model(model_cfg)

    data_cfg = cfg.data
    data = get_data(data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args)
    logger.info(f"----------Initializing Data...----------")
    logger.info(data)
    
    is_federated = cfg.trainer.handler in ["FederatedUnlearningTrainer", "FederatedFinetuneTrainer"]
    
    if is_federated:
        num_clients = cfg.trainer.method_args.get("num_clients", 3)
        target_client_idx = cfg.trainer.method_args.get("target_client_idx", 0)
        
        if mode == "unlearn":
            if "forget" not in data or "retain" not in data:
                raise ValueError("Both forget and retain data must be in data dictionary")
            federated_data = get_federated_data(
                data, num_clients=num_clients, target_client_idx=target_client_idx
            )

        # else:  # mode == "train" for federated fine-tuning
        #     if "train" not in data:
        #         raise ValueError("Training data must be in data dictionary")
        #     # 为联邦微调准备数据
        #     train_data = data["train"]
        #     federated_data = {}
        #     # 简单地将训练数据平均分配给各个客户端
        #     dataset_size = len(train_data)
        #     print(dataset_size)
        #     client_size = dataset_size // num_clients
        #     for i in range(num_clients):
        #         start_idx = i * client_size
        #         end_idx = start_idx + client_size if i < num_clients - 1 else dataset_size
        #         federated_data[i] = train_data[start_idx:end_idx]

        else:  # mode == "train" for federated fine-tuning
            if "train" not in data:
                raise ValueError("Training data must be in data dictionary")
            
            # 使用numpy高效处理索引
            train_data = data["train"]
            dataset_size = len(train_data)
            
            # 计算每个客户端的基础大小和额外数据
            base_size = dataset_size // num_clients
            extras = dataset_size % num_clients
            
            # 使用numpy创建并打乱索引
            indices = np.arange(dataset_size)
            rng = np.random.RandomState(cfg.trainer.args.seed)
            rng.shuffle(indices)
            
            # 使用Subset快速分配数据，避免数据复制
            federated_data = {}
            start_idx = 0
            for i in range(num_clients):
                # 计算当前客户端的数据大小（处理不能整除的情况）
                client_size = base_size + (1 if i < extras else 0)
                end_idx = start_idx + client_size
                
                # 使用Subset而不是切片，避免数据复制
                federated_data[i] = Subset(train_data, indices[start_idx:end_idx])
                start_idx = end_idx
        
        data["train"] = federated_data
        print(f"Prepared federated data for {num_clients} clients")

    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, "Please set trainer"

    evaluator = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        assert len(eval_cfgs) <= 1, ValueError("Only one evaluation supported while training")
        eval_name, eval_cfg = next(iter(eval_cfgs.items()))
        evaluator = get_evaluator(
            eval_name,
            eval_cfg,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluator=evaluator,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    # trainer_args.do_eval = False

    # if trainer_args.do_eval:
    #     trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()