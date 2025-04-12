import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluator
from trainer.utils import seed_everything

# python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=SAMPLE_UNLEARN

"""
   configs/
   ├── unlearn.yaml              # 基础配置
   ├── experiment/
   │   └── unlearn/
   │       └── tofu/
   │           └── default.yaml  # 实验配置
   ├── model/                    # 模型配置
   ├── trainer/                  # 训练器配置
   ├── data/                    # 数据配置
   └── eval/                    # 评估配置
"""


"""
Hydra配置系统:这个装饰器告诉Hydra: 1.配置文件在../configs目录下 2.默认使用train.yaml作为基础配置文件
Hydra会自动将这些命令行参数转换成配置对象, 这些参数会覆盖配置文件中的默认值, 所有这些配置最终会被合并到传递给main函数的cfg参数中
hydra会将最终的配置作为DictConfig对象传递给main函数: cfg
"""


"""
做联邦统一框架的大概逻辑:
1.设置3个客户端进行fine-tune, 数据进行切割/或者不进行这一步, 假设下载的集中式model就是联邦训练后的global model
2.得到一个global model, 并受到其中一个客户端的遗忘请求
3.定义这个客户端的遗忘数据, 并进行unlearning
4.将unlearning后的模型更新发送到服务器。服务器重新聚合模型,更新global model
5.在global model上进行评估,验证模型性能。
"""

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

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
