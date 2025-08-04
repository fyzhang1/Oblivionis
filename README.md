<!-- <div align="center">
<h1><strong>Oblivionis</strong></h2>
</div> -->

# Oblivionis: A Lightweight Learning and Unlearning Framework for Federated Large Language Models

---------
## 🔥 Introduction
We aim to construct a Lightweight Learning and Unlearning Framework for
Federated Large Language Models integrates federated fine-tuning and targeted unlearning, enabling robust LLM training while ensuring compliance with privacy regulations like GDPR. We support:

- Multiple FL algorithms
- Multiple Unlearning algorithms
- Multiple evaluation metrics
---------

## ⭐️ Setup
```python
git clone https://github.com/fyzhang1/Oblivionis.git
cd Oblivionis
conda create -n obl python=3.11
conda activate obl
pip install -r requirements.txt
python setup_data.py --eval
```

## 🏁 Quick Start
The training pipline script is in ```/scripts/main.sh```

```python
python src/federated_train.py \
  --config-name=train-lora.yaml \
  experiment=finetune/tofu/default.yaml \
  task_name=${finetune_task_name} \
  data/datasets@data.train=TOFU_QA_full \
  data.train.TOFU_QA_full.args.hf_args.name=full \
  paths.output_dir=saves/finetune/${finetune_task_name} \
  2>&1 | tee logs/finetune/${finetune_task_name}.txt
```
```python
python src/federated_train.py \
        --config-name=unlearn-lora.yaml \
        experiment=unlearn/tofu/default.yaml \
        forget_split=${forget_split} \
        retain_split=${retain_split} \
        trainer=FederatedUnlearningTrainer \
        task_name=${unlearn_task_name} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/${finetune_task_name} \
        retain_logs_path=saves/eval/${retain_task_name}/TOFU_EVAL.json \
        paths.output_dir=saves/unlearn/${unlearn_task_name}\
        2>&1 | tee logs/unlearn/${unlearn_task_name}.txt
```
Main config is in ```/configs/train-lora.yaml``` or ```/configs/unlearn-lora.yaml```
The Trainer configs are in ```/configs/trainer/FederatedFinetune.yaml``` and ```configs/trainer/FederatedUnlearningTrainer.yaml```
The experiment configs are in ```/configs/experiment```
The data configs are in ```/configs/data```


### 🔥 Lora

#### 1.model config

`configs/model/Llama-3.2-3B-Instruct-lora.yaml`  or create new file

```yaml
model_args:
  pretrained_model_name_or_path: "meta-llama/Llama-3.2-3B-Instruct"
  attn_implementation: 'flash_attention_2'
  torch_dtype: bfloat16
  use_lora: true
  lora_config:
    r: 32                   
    lora_alpha: 64       
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_dropout: 0.05       
    bias: "none"      
```

#### 2.training config

`configs/unlearn-lora.yaml`:

```yaml
defaults:
  - model: Llama-3.2-3B-Instruct-lora
  - trainer: FederatedUnlearningTrainer
  - data: unlearn
  - collator: DataCollatorForSupervisedDataset
  - eval: tofu
  - hydra: default
  - paths: default
  - experiment: null
  - _self_

trainer:
  args: 
    remove_unused_columns: False

mode: unlearn
task_name: ???
```


