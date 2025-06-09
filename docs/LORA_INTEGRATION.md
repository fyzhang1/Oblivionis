# LoRA集成指南

本文档介绍如何在联邦学习LLM遗忘项目中使用LoRA (Low-Rank Adaptation) 技术。

## 什么是LoRA

LoRA (Low-Rank Adaptation) 是一种参数高效的微调技术，它通过在预训练模型的线性层中添加低秩矩阵来实现模型适应，而不需要更新原始模型的全部参数。

### LoRA的优势

1. **内存高效**: 只需要训练少量的额外参数（通常是原模型的0.1%-10%）
2. **存储高效**: LoRA适配器文件很小，易于存储和传输
3. **联邦学习友好**: 减少客户端之间需要传输的参数数量
4. **快速切换**: 可以快速加载不同的LoRA适配器

## 安装依赖

确保安装了peft库：

```bash
pip install peft==0.13.2
```

或者从requirements.txt安装：

```bash
pip install -r requirements.txt
```

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

### 2. 使用示例脚本

```bash
python scripts/run_federated_lora_unlearn.py \
  --config-name=unlearn-lora.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 \
  retain_split=retain90 \
  task_name=fed_lora_unlearn \
  model.model_args.pretrained_model_name_or_path=saves/finetune/SAMPLE_TRAIN
```

### 3. 加载已有的LoRA模型

如果你已经有一个训练好的LoRA模型，可以这样加载：

```bash
python src/fed_train.py \
  --config-name=unlearn-lora.yaml \
  model.model_args.lora_model_path=path/to/your/lora/model \
  # ... 其他参数
```

## LoRA参数说明

### 核心参数

- **r (rank)**: LoRA矩阵的秩，控制适配器的复杂度
  - 较小的值（4-16）：参数更少，训练更快，但表达能力有限
  - 较大的值（32-64）：表达能力更强，但参数更多

- **lora_alpha**: 控制LoRA层输出的缩放
  - 通常设置为rank的2倍
  - 影响LoRA适配器对原模型的影响程度

- **target_modules**: 要应用LoRA的模块
  - 对于Llama模型，通常包括attention和MLP层
  - 可以选择性地只对某些层应用LoRA以节省参数

- **lora_dropout**: LoRA层的dropout率
  - 用于防止过拟合
  - 通常设置为0.05-0.1

### 推荐配置

#### 快速原型（最小参数）
```yaml
lora_config:
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "v_proj"]
  lora_dropout: 0.1
```

#### 平衡配置（推荐）
```yaml
lora_config:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.1
```

#### 高性能配置
```yaml
lora_config:
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.05
```

## 联邦学习中的LoRA

在联邦学习场景中，LoRA有以下优势：

1. **通信效率**: 只需要传输LoRA适配器的权重，大大减少通信开销
2. **隐私保护**: 基础模型参数不变，只传输适配器权重
3. **客户端资源友好**: 每个客户端只需要存储和计算LoRA参数

### 聚合策略

项目支持多种联邦聚合策略与LoRA结合：

- **FedAvg**: 标准联邦平均
- **FedAvgM**: 带动量的联邦平均
- **FedAdagrad**: 自适应学习率
- **FedAdam**: Adam优化器的联邦版本
- **FedYogi**: Yogi优化器的联邦版本
- **FedProx**: 近端联邦优化

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小LoRA rank (r)
   - 减少target_modules
   - 调整batch size

2. **训练不收敛**
   - 增加LoRA rank
   - 调整lora_alpha
   - 检查学习率设置

3. **模型保存/加载问题**
   - 确保使用`save_pretrained()`保存PEFT模型
   - 使用`PeftModel.from_pretrained()`加载

### 性能优化

1. **内存优化**:
   ```yaml
   model_args:
     torch_dtype: bfloat16  # 使用半精度
     use_lora: true
     lora_config:
       r: 8  # 较小的rank
   ```

2. **速度优化**:
   ```yaml
   trainer:
     args:
       gradient_checkpointing: true
       dataloader_num_workers: 4
   ```

## 实验建议

1. **超参数搜索**: 从小的rank开始，逐步增加直到性能满足需求
2. **目标模块选择**: 可以通过实验确定哪些模块对任务最重要
3. **联邦设置**: 在LoRA设置下可能需要调整联邦学习的超参数

## 参考资料

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [PEFT库文档](https://huggingface.co/docs/peft)
- [联邦学习与LoRA结合的研究](https://arxiv.org/abs/2302.08888) 