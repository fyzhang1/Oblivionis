<div align="center">
<h2><strong>Federated LLM Unlearning</strong></h2>
</div>

---------
## Introduction
We aim to construct the Federated LLM Unlearning: A unified framework for building secure federated large models that are trainable and forgettable on local private data.

- Multiple federated learning algorithms
- Multiple evaluation indicators
- Multiple unlearning and federated unlearning algorithms
---------
### Quick Start
##### Fine-Tune (Temporarily use centralized training to fine-tune a global model)
```python
python src/train.py --config-name=train.yaml experiment=finetune/tofu/default task_name=SAMPLE_TRAIN
```
```python
The fine-tune model (initial global model) is saved: "saves/finetune/SAMPLE_TRAIN"
```
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

##### Eval: Using TOFU benchmark to evaluate global model
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

---------
## Acknowledgements

- This repo is inspired from [OpenUnlearning](https://github.com/locuslab/open-unlearning). 

---------------------------
