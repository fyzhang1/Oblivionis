#!/usr/bin/env python3
"""
联邦学习LoRA遗忘训练示例脚本

这个脚本展示了如何使用LoRA技术进行联邦学习的LLM遗忘训练。
LoRA (Low-Rank Adaptation) 可以显著减少训练参数数量和内存使用。

使用示例:
python scripts/run_federated_lora_unlearn.py \
    --config-name=unlearn-lora.yaml \
    experiment=unlearn/tofu/default \
    forget_split=forget10 \
    retain_split=retain90 \
    trainer=FederatedUnlearningTrainer \
    task_name=fed_lora_unlearn \
    model=Llama-3.2-3B-Instruct-lora \
    model.model_args.pretrained_model_name_or_path=saves/finetune/SAMPLE_TRAIN \
    trainer.method_args.num_clients=3 \
    trainer.method_args.target_client_idx=0 \
    trainer.method_args.global_rounds=5 \
    trainer.method_args.aggregation_strategy=FedAvg
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import hydra
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="unlearn-lora.yaml")
def main(cfg: DictConfig):
    # 导入需要的模块
    from fed_train import main as fed_main
    
    # 打印配置信息
    logger.info("=" * 60)
    logger.info("联邦学习LoRA遗忘训练")
    logger.info("=" * 60)
    logger.info(f"模型: {cfg.get('model', {}).get('_target_', 'Llama-3.2-3B-Instruct-lora')}")
    logger.info(f"任务名称: {cfg.get('task_name', 'Unknown')}")
    
    # 安全地访问trainer配置
    trainer_cfg = cfg.get('trainer', {})
    method_args = trainer_cfg.get('method_args', {})
    
    logger.info(f"客户端数量: {method_args.get('num_clients', 3)}")
    logger.info(f"目标客户端: {method_args.get('target_client_idx', 0)}")
    logger.info(f"全局轮数: {method_args.get('global_rounds', 1)}")
    logger.info(f"聚合策略: {method_args.get('aggregation_strategy', 'FedAvg')}")
    
    # LoRA特定配置
    model_cfg = cfg.get('model', {})
    model_args = model_cfg.get('model_args', {})
    if model_args.get('use_lora', False):
        lora_config = model_args.get('lora_config', {})
        logger.info("LoRA配置:")
        logger.info(f"  - rank (r): {lora_config.get('r', 16)}")
        logger.info(f"  - alpha: {lora_config.get('lora_alpha', 32)}")
        logger.info(f"  - dropout: {lora_config.get('lora_dropout', 0.1)}")
        logger.info(f"  - target_modules: {lora_config.get('target_modules', [])}")
    else:
        logger.info("未启用LoRA配置")
    
    logger.info("=" * 60)
    
    # 调用原始的fed_train主函数
    fed_main(cfg)

if __name__ == "__main__":
    main() 