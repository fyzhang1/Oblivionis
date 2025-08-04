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
    

     # Print training configuration
    logger.info(f"-----Training Configuration-----")
    if hasattr(cfg.trainer, 'args'):
        trainer_args = cfg.trainer.args
        logger.info(f"Learning Rate: {getattr(trainer_args, 'learning_rate', 'N/A')}")
        logger.info(f"Batch Size: {getattr(trainer_args, 'per_device_train_batch_size', 'N/A')}")
        logger.info(f"Num Epochs: {getattr(trainer_args, 'num_train_epochs', 'N/A')}")
        logger.info(f"Warmup Steps: {getattr(trainer_args, 'warmup_steps', 'N/A')}")
        logger.info(f"Seed: {getattr(trainer_args, 'seed', 'N/A')}")
        logger.info(f"Output Dir: {getattr(trainer_args, 'output_dir', 'N/A')}")
        logger.info(f"Gradient Accumulation Steps: {getattr(trainer_args, 'gradient_accumulation_steps', 'N/A')}")
        logger.info(f"Weight Decay: {getattr(trainer_args, 'weight_decay', 'N/A')}")
        logger.info(f"LR Scheduler Type: {getattr(trainer_args, 'lr_scheduler_type', 'N/A')}")
    logger.info(f"------------------------------")

    # 添加调试信息
    logger.info(f"Model Configuration: {model_cfg}")
    logger.info(f"Model Args: {model_cfg.model_args}")
    logger.info(f"Use LoRA: {model_cfg.model_args.get('use_lora', 'NOT_FOUND')}")
    logger.info(f"LoRA Configuration: {model_cfg.model_args.get('lora_config', 'NOT_FOUND')}")
    
    model, tokenizer = get_model(model_cfg)

    # PEFT信息
    if hasattr(model, "peft_config"):
        adapter_names = list(model.peft_config.keys())
        logger.info(f"Loaded adapters: {adapter_names}")

    data_cfg = cfg.data
    data = get_data(data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args)
    logger.info(f"----------Initializing Data----------")
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

        else:  # mode == "train" for federated fine-tuning
            if "train" not in data:
                raise ValueError("Training data must be in data dictionary")
            
          
            train_data = data["train"]
            dataset_size = len(train_data)
            
       
            base_size = dataset_size // num_clients
            extras = dataset_size % num_clients
            
      
            indices = np.arange(dataset_size)
            rng = np.random.RandomState(cfg.trainer.args.seed)
            rng.shuffle(indices)
            
      
            federated_data = {}
            start_idx = 0
            for i in range(num_clients):
      
                client_size = base_size + (1 if i < extras else 0)
                end_idx = start_idx + client_size
                
 
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