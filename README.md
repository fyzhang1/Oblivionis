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
- Training: Federated Unlearning (GradAscent, Fedavg)
```python
src/fed_train.py --config-name=unlearn.yaml             
experiment=unlearn/tofu/default \forget_split=forget10          
retain_split=retain90 trainer=FederatedUnlearningTrainer task_name=test
```
```python
The global model is saved "saves/unlearn/test" #test is the task_name
```
- Eval: Using TOFU benchmark to evaluate global model
```python
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
experiment=eval/tofu/default.yaml \
forget_split=forget10 \
holdout_split=retain90 \
task_name=test \
model.model_args.pretrained_model_name_or_path=saves/unlearn/test \
paths.output_dir=saves/unlearn/test/evals \
retain_logs_path=saves/eval/test/TOFU_EVAL.json
```

```python
The results are saved "saves/unlearn/test/evals" 
```
---------
## Acknowledgements

- This repo is inspired from [OpenUnlearning](https://github.com/locuslab/open-unlearning). 

---------------------------
