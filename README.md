## 配置文件

### 1. 模型配置

创建或使用`configs/model/Llama-3.2-3B-Instruct-lora.yaml`:

```yaml
model_args:
  pretrained_model_name_or_path: "meta-llama/Llama-3.2-3B-Instruct"
  attn_implementation: 'flash_attention_2'
  torch_dtype: bfloat16
  use_lora: true
  lora_config:
    r: 16                    # LoRA rank
    lora_alpha: 32          # LoRA alpha参数
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_dropout: 0.1       # LoRA dropout率
    bias: "none"            # bias处理方式
```

### 2. 训练配置

使用`configs/unlearn-lora.yaml`:

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

## 使用方法

### 1. 基本使用

```bash
python src/fed_train.py \
  --config-name=unlearn-lora.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 \
  retain_split=retain90 \
  trainer=FederatedUnlearningTrainer \
  task_name=fed_lora_unlearn \
  model=Llama-3.2-3B-Instruct-lora \
  model.model_args.pretrained_model_name_or_path=saves/finetune/SAMPLE_TRAIN
```