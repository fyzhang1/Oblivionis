#!/bin/bash
set -ex

# ========================================
#     Parameters Configuration
# ========================================
forget_split_list=("forget01" "forget05" "forget10")
retain_split_list=("retain99" "retain95" "retain90")
holdout_split_list=("holdout01" "holdout05" "holdout10")

FL_Method="FedAvg"
Model_Size="1B"

# ========================================
#     Log/Save Directory Preparation
# ========================================
mkdir -p logs/retain/eval
mkdir -p logs/finetune/eval
mkdir -p logs/unlearn/eval

mkdir -p saves/retain
mkdir -p saves/finetune
mkdir -p saves/unlearn
mkdir -p saves/eval

# ========================================
#     Warning
# ========================================
num_splits=${#forget_split_list[@]}
read -p "Ready to run the whole experiments? [Y/n] " confirm
[[ $confirm == [Nn]* ]] && exit 1

# ========================================
#     Retain
# ========================================
declare -a retain_task_names

for ((i=0; i<$num_splits; i++)); do
  forget_split=${forget_split_list[$i]}
  retain_split=${retain_split_list[$i]}
  holdout_split=${holdout_split_list[$i]}

  retain_task_name="${Model_Size}_${FL_Method}_Retain_${retain_split}"
  retain_task_names[$i]=$retain_task_name

  echo "[INFO] Retain Training: ${retain_split} (${retain_task_name})"

  python src/federated_train.py \
    --config-name=train-lora.yaml \
    experiment=finetune/tofu/default.yaml \
    task_name=${retain_task_name} \
    data/datasets@data.train=TOFU_QA_retain \
    data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
    paths.output_dir=saves/retain/${retain_task_name} \
    2>&1 | tee logs/retain/${retain_task_name}.txt

  echo "[Step 1 ✅] Retain Train Finish：${retain_split}"

  python src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=${forget_split} \
    holdout_split=${holdout_split} \
    task_name=${retain_task_name} \
    model.model_args.pretrained_model_name_or_path=saves/finetune/${retain_task_name} \
    paths.output_dir=saves/eval/${retain_task_name} \
    hydra.run.dir=saves/eval/${retain_task_name} \
    2>&1 | tee logs/retain/eval/${retain_task_name}.txt

  echo "[Step 2 ✅] Retain Eval Finish：${retain_split}"
done

# ========================================
#     Finetune
# ========================================
finetune_task_name="${Model_Size}_${FL_Method}_Finetune"

python src/federated_train.py \
  --config-name=train-lora.yaml \
  experiment=finetune/tofu/default.yaml \
  task_name=${finetune_task_name} \
  data/datasets@data.train=TOFU_QA_full \
  data.train.TOFU_QA_full.args.hf_args.name=full \
  paths.output_dir=saves/finetune/${finetune_task_name} \
  2>&1 | tee logs/finetune/${finetune_task_name}.txt

echo "[Step 3 ✅] Finetune Finish"

for ((i=0; i<$num_splits; i++)); do
  forget_split=${forget_split_list[$i]}
  retain_task_name=${retain_task_names[$i]}
  holdout_split=${holdout_split_list[$i]}

  echo "[INFO] i=${i} | forget=${forget_split} | retain_task=${retain_task_name} | holdout=${holdout_split}"

  python src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=${forget_split} \
    holdout_split=${holdout_split} \
    task_name=${finetune_task_name} \
    model.model_args.pretrained_model_name_or_path=saves/finetune/${finetune_task_name} \
    retain_logs_path=saves/eval/${retain_task_name}/TOFU_EVAL.json \
    paths.output_dir=saves/eval/${finetune_task_name}/${forget_split} \
    hydra.run.dir=saves/eval/${finetune_task_name}/${forget_split} \
    2>&1 | tee logs/finetune/eval/${finetune_task_name}_${forget_split}.txt

  echo "[Step 4 ✅] Finetune Eval Finish：forget=${forget_split}"
done

# ========================================
#     Unlearning
# ========================================
unlearn_cls=("GradAscent" "GradDiff" "NPO" "SimNPO")

default_yaml="configs/experiment/unlearn/tofu/default.yaml"
trainer_yaml="configs/trainer/FederatedUnlearningTrainer.yaml"

for AfterUnlearn in "${unlearn_cls[@]}"; do
  echo -e "\n==============================="
  echo ">>> Present Unlearn Method: $AfterUnlearn"
  echo "==============================="

  trainer_line=$(grep '\- override /trainer:' "$default_yaml")
  if [[ -z "$trainer_line" ]]; then
    echo "❌ Error: Unable to find override /trainer line"
    exit 1
  fi
  BeforeUnlearn=$(echo "$trainer_line" | awk -F": " '{print $2}' | xargs)

  current_cls=$(grep "^[[:space:]]*unlearn_trainer_cls:" "$trainer_yaml" | awk -F'"' '{print $2}' | xargs)

  if [[ "$BeforeUnlearn" == "$current_cls" ]]; then
    echo "[ACTION] Replace Trainer：$BeforeUnlearn → $AfterUnlearn"

    sed -i "s|^\([[:space:]]*- override /trainer:\).*|\1 $AfterUnlearn|" "$default_yaml"
    sed -i "s|^\([[:space:]]*unlearn_trainer_cls:\).*|\1 \"$AfterUnlearn\"|" "$trainer_yaml"

    for ((i=0; i<$num_splits; i++)); do
      forget_split=${forget_split_list[$i]}
      retain_split=${retain_split_list[$i]}
      holdout_split=${holdout_split_list[$i]}
      retain_task_name=${retain_task_names[$i]}

      unlearn_task_name=${Model_Size}_${FL_Method}_${AfterUnlearn}_Unlearn_${forget_split}

      # Step 5: Unlearn
      echo "[INFO] Unlearn Train: $unlearn_task_name"

      python src/federated_train.py \
        --config-name=unlearn-lora.yaml \
        experiment=unlearn/tofu/default.yaml \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        trainer=FederatedUnlearningTrainer \
        task_name=${unlearn_task_name} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/${finetune_task_name} \
        retain_logs_path=saves/eval/${retain_task_name}/TOFU_EVAL.json \
        paths.output_dir=saves/unlearn/${unlearn_task_name} \
        2>&1 | tee logs/unlearn/${unlearn_task_name}.txt

      echo "[Step 5 ✅] Unlearn Train Finish：$unlearn_task_name"

      # Step 6: Unlearn Eval
      python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=${unlearn_task_name} \
        retain_logs_path=saves/eval/${retain_task_name}/TOFU_EVAL.json \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${unlearn_task_name} \
        hydra.run.dir=saves/eval/${unlearn_task_name} \
        paths.output_dir=saves/eval/${unlearn_task_name} \
        2>&1 | tee logs/unlearn/eval/${unlearn_task_name}.txt

      echo "[Step 6 ✅] Unlearn Eval Finish：$unlearn_task_name"
    done
  else
    echo "⚠️ Config Mismatch（$BeforeUnlearn ≠ $current_cls），Skip $AfterUnlearn"
  fi
done

echo -e "\n✅ All experiments completed! ✅"