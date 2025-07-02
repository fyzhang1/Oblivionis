<div align="center">
<h1><strong>🌓 Federated LLM Unlearning</strong></h2>
</div>


---------
## 🔥 Introduction
We aim to construct the Federated LLM Unlearning: A unified framework for building secure federated large models that are trainable and forgettable on local private data.

- Multiple federated learning algorithms
- Multiple evaluation indicators
- Multiple unlearning and federated unlearning algorithms
---------
# 🏁 Quick Start
##  🧩 LoRA Federated Finetune (TOFU)
```python
python src/federated_train.py \
--config-name=train-lora.yaml \
experiment=finetune/tofu/default \
task_name=FedProx_LoRA_Finetune_llama3b
```
`task_name={fed}_LoRA+{Finetune\Unlearn name}_{model}`

`The fine-tune model (initial global model) is saved: "saves/finetune/{task_name}"`


##  🪭TOFU Unlearning

##### Unlearning: Federated Unlearning (GradAscent, Fedavg)
```python
python src/fed_train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \forget_split=forget10 retain_split=retain90 trainer=FederatedUnlearningTrainer task_name=test \
model=Llama-3.2-3B-Instruct \
model.model_args.pretrained_model_name_or_path=saves/finetune/SAMPLE_TRAIN
```
```python
The global model is saved "saves/unlearn/test" #test is the task_name
```
- config file: "configs/trainer/FederatedUnlearningTrainer"

## TOFU Eval

##### Using TOFU benchmark to evaluate federated global model

```python
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
experiment=eval/tofu/default.yaml \
forget_split=forget10 \
holdout_split=retain90 \
task_name=test \
model.model_args.pretrained_model_name_or_path=saves/unlearn/test \
paths.output_dir=saves/unlearn/test/evals \
retain_logs_path=saves/eval/SAMPLE_TRAIN/TOFU_EVAL.json
```

```python
The results are saved "saves/unlearn/test/evals" 
```
- retain_logs_path: the reference of initial global's evaluation


## 🔑 MUSE Unlearning
##### MUSE: federated training (A100 80G)
```python
python src/fed_train.py --config-name=unlearn.yaml \
experiment=unlearn/muse/default.yaml \
model=Llama-2-7b-hf \
data_split=News \
trainer=FederatedUnlearningTrainer \
task_name=test \
retain_logs_path=saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json \
trainer.args.per_device_train_batch_size=2 \
trainer.args.gradient_accumulation_steps=8
```

##  🚀 Support

### 🔥❄️ Lora

#### 1.model config

`configs/model/Llama-3.2-3B-Instruct-lora.yaml`  or create new file

```yaml
model_args:
  pretrained_model_name_or_path: "meta-llama/Llama-3.2-3B-Instruct"
  attn_implementation: 'flash_attention_2'
  torch_dtype: bfloat16
  use_lora: true
  lora_config:
    r: 32                    # LoRA rank
    lora_alpha: 64          # LoRA alpha参数
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_dropout: 0.05       # LoRA dropout率
    bias: "none"            # bias处理方式
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



#### Start Lora

```bash
python src/fed_train.py \
  --config-name=unlearn-lora.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 \
  retain_split=retain90 \
  trainer=FederatedUnlearningTrainer \
  task_name=fed_lora_unlearn \
  model=Llama-3.2-3B-Instruct-lora \ #1B/3B/7B
  model.model_args.pretrained_model_name_or_path=saves/finetune/SAMPLE_TRAIN #change
```



## ⭐️ Acknowledgements

- This repo is inspired from [OpenUnlearning](https://github.com/locuslab/open-unlearning). 

---------------------------


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
