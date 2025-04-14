import hydra
from omegaconf import DictConfig
from data import get_data, get_collators, get_federated_data
from model import get_model
from trainer import load_trainer
from evals import get_evaluator
from trainer.utils import seed_everything


# python src/fed_train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \forget_split=forget10 retain_split=retain90 trainer=FederatedUnlearningTrainer task_name=test

"""
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port 18765 \
  src/fed_train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \forget_split=forget10 retain_split=retain90 trainer=FederatedUnlearningTrainer task_name=test


# python src/eval.py  experiment=eval/tofu/default.yaml task_name=test

  

            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=forget10 \
            holdout_split=retain90 \
            task_name=test \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/test \
            paths.output_dir=saves/unlearn/test/evals \
            retain_logs_path=saves/eval/test/TOFU_EVAL.json

"""
@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    data_cfg = cfg.data
    data = get_data(data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args)
    print("-------------------------------------------------init data----------------------------------")
    print(data)
    # print(f"forget_data type: {type(data.get("forget"))}")
    is_federated = cfg.trainer.handler == "FederatedUnlearningTrainer"
    
    if is_federated and mode == "unlearn":
        num_clients = cfg.trainer.method_args.get("num_clients", 3)
        target_client_idx = cfg.trainer.method_args.get("target_client_idx", 0)
        
        if "forget" not in data or "retain" not in data:
            raise ValueError("Both forget and retain data must be in data dictionary")
                
        federated_data = get_federated_data(
            data, num_clients=num_clients, target_client_idx=target_client_idx
        )
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
            template_args=template_args,  # 明确作为关键字参数
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