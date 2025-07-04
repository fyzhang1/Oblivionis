#!/bin/bash
set -ex

# === Step 0: 先执行 Finetune 和 Eval ===
python src/federated_train.py \
  --config-name=train-lora.yaml \
  experiment=finetune/tofu/default \
  task_name=FedAvg_LoRA_Finetune \
  2>&1 | tee logs/FedAvg_LoRA_Finetune.txt

python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=forget10 \
  holdout_split=holdout10 \
  task_name=FedAvg_LoRA_Finetune_Eval \
  model.model_args.pretrained_model_name_or_path=saves/finetune/FedAvg_LoRA_Finetune \
  2>&1 | tee logs/FedAvg_LoRA_Finetune_Eval.txt

# === Step 1: 定义所有 Unlearning 方法（顺序执行） ===
unlearn_cls=(
  "GradAscent"
  "GradDiff"
  "NPO"
  "SimNPO"
)

# === Step 2: 配置文件路径 ===
default_yaml="configs/experiment/unlearn/tofu/default.yaml"
trainer_yaml="configs/trainer/FederatedUnlearningTrainer.yaml"

# === Step 3: 顺序尝试每个方法 ===
for AfterUnlearn in "${unlearn_cls[@]}"; do
  echo -e "\n==============================="
  echo ">>> 当前目标方法: $AfterUnlearn"
  echo "==============================="

  # 提取 default.yaml 中 override /trainer 的方法（允许前面有空格）
  trainer_line=$(grep '\- override /trainer:' "$default_yaml")
  if [[ -z "$trainer_line" ]]; then
    echo "❌ 错误：未找到 override /trainer 行"
    exit 1
  fi
  BeforeUnlearn=$(echo "$trainer_line" | awk -F": " '{print $2}' | xargs)  # xargs去掉空格
  echo "[INFO] default.yaml 中 BeforeUnlearn: $BeforeUnlearn"

  # 提取 FederatedUnlearningTrainer.yaml 中的 unlearn_trainer_cls
  current_cls=$(grep "^[[:space:]]*unlearn_trainer_cls:" "$trainer_yaml" | awk -F'"' '{print $2}' | xargs)
  echo "[INFO] FederatedUnlearningTrainer.yaml 中 unlearn_trainer_cls: $current_cls"

  # 如果两个配置一致，则替换为 AfterUnlearn
  if [[ "$BeforeUnlearn" == "$current_cls" ]]; then
    echo "[ACTION] 配置一致，执行替换为 $AfterUnlearn"

    # 替换 default.yaml 中 override /trainer
    sed -i "s|^\([[:space:]]*- override /trainer:\).*|\1 $AfterUnlearn|" "$default_yaml"

    # 替换 trainer_yaml 中的 unlearn_trainer_cls（带引号）
    sed -i "s|^\([[:space:]]*unlearn_trainer_cls:\).*|\1 \"$AfterUnlearn\"|" "$trainer_yaml"

    echo "✅ 替换完成：$BeforeUnlearn → $AfterUnlearn"

    # === Step 4: Unlearning 训练 ===
    python src/federated_train.py \
      --config-name=unlearn-lora.yaml \
      experiment=unlearn/tofu/default \
      forget_split=forget10 \
      retain_split=retain90 \
      trainer=FederatedUnlearningTrainer \
      task_name=FedAvg_LoRA_${AfterUnlearn}_Unlearn \
      model.model_args.pretrained_model_name_or_path=saves/finetune/FedAvg_LoRA_Finetune \
      2>&1 | tee logs/FedAvg_LoRA_${AfterUnlearn}_Unlearn.txt

    # === Step 5: Unlearning 评估 ===
    python src/eval.py \
      experiment=eval/tofu/default.yaml \
      forget_split=forget10 \
      holdout_split=holdout10 \
      task_name=FedAvg_LoRA_${AfterUnlearn}_Unlearn_Eval \
      retain_logs_path=saves/eval/FedAvg_LoRA_Finetune_Eval/TOFU_EVAL.json \
      model.model_args.pretrained_model_name_or_path=saves/unlearn/FedAvg_LoRA_${AfterUnlearn}_Unlearn \
      2>&1 | tee logs/FedAvg_LoRA_${AfterUnlearn}_Unlearn_Eval.txt
  else
    echo "⚠️ 配置不一致（$BeforeUnlearn ≠ $current_cls），跳过 $AfterUnlearn"
  fi
done

echo -e "\n✅ 全部运行完成。"