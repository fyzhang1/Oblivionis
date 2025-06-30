import hydra
from data import get_data, get_collators
from evals import get_evaluator
from model import get_model
from omegaconf import DictConfig
from trainer import load_trainer
from trainer.utils import seed_everything
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    # 只打印训练相关的参数
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

    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Addtional logging for model configuration
    logger.info(f"Model Configuration: {model_cfg}")
    logger.info(f"Model Args: {model_cfg.model_args}")
    logger.info(f"Use LoRA: {model_cfg.model_args.get('use_lora', 'NOT_FOUND')}")
    logger.info(f"LoRA Configuration: {model_cfg.model_args.get('lora_config', 'NOT_FOUND')}")

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluator
    evaluator = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        assert len(eval_cfgs) <= 1, ValueError(
            "Only one evaluation supported while training"
        )
        eval_name, eval_cfg = next(iter(eval_cfgs.items()))
        evaluator = get_evaluator(
            eval_name,
            eval_cfg,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    evaluator = None
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

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()