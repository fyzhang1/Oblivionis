<h1 align="center">
  <img src="./assets/icon.svg" alt="logo" width="42" />Oblivionis: A Lightweight Learning and Unlearning Framework for Federated Large Language Models
</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2508.08875">
    <img src="https://img.shields.io/badge/arXiv-2508.08875-b31b1b.svg?style=for-the-badge&logo=arxiv&logoWidth=20" alt="arXiv"></a>
  &nbsp;&nbsp;
  <a href="https://fyzhang1.github.io/Oblivionis/"><img src="https://img.shields.io/badge/Project-Page-779977?style=for-the-badge&logoWidth=20"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/fyzhang1/Oblivionis/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/LICENSE-MIT-7799CC.svg?style=for-the-badge&logoWidth=20" alt="License MIT"></a>
</p>
<p align="center">
  <a href="https://github.com/fyzhang1">Fuyao Zhang</a><sup>1,*</sup>, 
  <a href="https://github.com/Tristan-SHU">Xinyu Yan</a><sup>1,*</sup>, 
  <a href="https://scholar.google.com/citations?user=7YsN6lMAAAAJ">Tiantong Wu</a><sup>1</sup>, 
  <a href="https://imamtom.github.io/">Wenjie Li</a><sup>1,2</sup>, 
  Tianxiang Chen</a><sup>1,3</sup>,<br>
  <a href="https://yangcao888.github.io/">Yang Cao</a><sup>4</sup>, 
  <a href="https://ranyan.w3spaces.com/">Ran Yan</a><sup>1</sup>, 
  <a href="https://scholar.google.com/citations?user=EQDfV9cAAAAJ">Longtao Huang</a><sup>5</sup>, 
  <a href="https://sites.google.com/view/wyb">Wei Yang Bryan Lim</a><sup>1,†</sup>, 
  <a href="https://www.polyu.edu.hk/dsai/docdrive/personal/yangqiang.html">Qiang Yang</a><sup>6</sup>
</p>

<p align="center">
  <sup>1 </sup>Nanyang Technological University
  <sup>2 </sup>Hebei Normal University
  <sup>3 </sup>Beihang University<br>
  <sup>4 </sup>Institute of Science Tokyo
  <sup>5 </sup>Alibaba Group
  <sup>6 </sup>Hong Kong Polytechnic University
</p>


---

> **「我、忘却を恐れるなかれ」 — 豊川祥子**


This repo is the official implementation of **Oblivionis** proposed by our paper ["Oblivionis: A Lightweight Learning and Unlearning Framework for Federated Large Language Models"](https://arxiv.org/abs/2508.08875).

**Oblivionis** is a Python framework for the reproduction and development of the learning and unlearning of Federated Large Language Models research. It integrates 6 representative federated learning algorithms, 5 machine unlearning methods, 2 federated fine-tuning methods (full-parameter and LoRA-based), and a variety of models. **Oblivionis** also supports 5 datasets and over 10 evaluation metrics.

---

## 🗺️ Navigation

- [📖 Introduction](#-introduction)
- [📃 Changelog](#-changelog)
- [⚡ Quick Start](#-quick-start)
- [🧪 Experiments](#-experiments)
  - [📚 Retain](#-retain)
  - [🎯 Finetune](#-finetune)
  - [✂️ Unlearn](#-unlearn)
  - [📊 Evaluation](#-evaluation)
  - [🧭 Baseline](#-baseline)

- [🔎 Roadmap](#-roadmap)
- [🤝 Acknowledgment](#-acknowledgment)
- [🔖 License](#-license)
- [📄 Citation](#-citation)

## 📖 Introduction

Large Language Models (LLMs) increasingly leverage Federated Learning (FL) to utilize private, task-specific datasets for fine-tuning while preserving data privacy. However, while federated LLM frameworks effectively enable collaborative training without raw data sharing, they critically lack built-in mechanisms for regulatory compliance. Integrating private data heightens concerns over data quality and long-term governance, yet existing distributed training frameworks offer no principled way to selectively remove specific client contributions post-training. Due to distributed data silos, stringent privacy constraints, and the intricacies of interdependent model aggregation, federated LLM unlearning is significantly more complex than centralized LLM unlearning.

To address this gap, we introduce **Oblivionis**, a lightweight learning and unlearning framework that enables clients to selectively remove specific private data during federated LLM training, enhancing trustworthiness and regulatory compliance.

![overview](./assets/overview.png)

<p align="center">Figure 1. Overview of the proposed <b>Oblivionis</b> framework.
</p>


## 📃 Changelog

- **[2025-08-13]** The preprint version of our paper was published on [arXiv](https://arxiv.org/abs/2508.08875).
- **[2025-08-04]** We released the first version of **Oblivionis** on GitHub.

## ⚡ Quick Start

To get started with **Oblivionis**, you can simply install it with git, conda and pip:

```shell
# Download
git clone https://github.com/fyzhang1/Oblivionis.git
cd Oblivionis

# Set up & Install
conda create -n Oblivionis python=3.12
conda activate Oblivionis
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 🧪 Experiments

Here, we use the default configuration file with the **TOFU** benchmark as the example. If you need to customize the experiment for your research, please modify the following files:

- **Main Configurations**: `configs/train-lora.yaml` and `configs/unlearn-lora.yaml`
- **Trainer Configurations**: `configs/trainer/FederatedFinetune.yaml` and `configs/trainer/FederatedUnlearningTrainer.yaml`
- **Experiment Settings**: `configs/experiment/`
- **Datasets**: `configs/data/`

### 📚 Retain

```shell
python src/federated_train.py \
  --config-name=train-lora.yaml \
  experiment=finetune/tofu/default.yaml \
  task_name=${retain_task_name} \
  data/datasets@data.train=TOFU_QA_retain \
  data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
  paths.output_dir=saves/retain/${retain_task_name}
```

### 🎯 Finetune

```shell
python src/federated_train.py \
  --config-name=train-lora.yaml \
  experiment=finetune/tofu/default.yaml \
  task_name=${finetune_task_name} \
  data/datasets@data.train=TOFU_QA_full \
  data.train.TOFU_QA_full.args.hf_args.name=full \
  paths.output_dir=saves/finetune/${finetune_task_name}
```

### ✂️ Unlearn

```shell
python src/federated_train.py \
  --config-name=unlearn-lora.yaml \
  experiment=unlearn/tofu/default.yaml \
  forget_split=${forget_split} \
  retain_split=${retain_split} \
  trainer=FederatedUnlearningTrainer \
  task_name=${unlearn_task_name} \
  model.model_args.pretrained_model_name_or_path=saves/finetune/${finetune_task_name} \
  retain_logs_path=saves/eval/${retain_task_name}/TOFU_EVAL.json \
  paths.output_dir=saves/unlearn/${unlearn_task_name}
```

### 📊 Evaluation

```shell
# Retain
python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=${forget_split} \
  holdout_split=${holdout_split} \
  task_name=${retain_task_name} \
  model.model_args.pretrained_model_name_or_path=saves/finetune/${retain_task_name} \
  paths.output_dir=saves/eval/${retain_task_name} \
  hydra.run.dir=saves/eval/${retain_task_name}

# Finetune
python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=${forget_split} \
  holdout_split=${holdout_split} \
  task_name=${finetune_task_name} \
  model.model_args.pretrained_model_name_or_path=saves/finetune/${finetune_task_name} \
  retain_logs_path=saves/eval/${retain_task_name}/TOFU_EVAL.json \
  paths.output_dir=saves/eval/${finetune_task_name}/${forget_split} \
  hydra.run.dir=saves/eval/${finetune_task_name}/${forget_split}

# Unlearn
python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=${forget_split} \
  holdout_split=${holdout_split} \
  task_name=${unlearn_task_name} \
  retain_logs_path=saves/eval/${retain_task_name}/TOFU_EVAL.json \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/${unlearn_task_name} \
  hydra.run.dir=saves/eval/${unlearn_task_name} \
  paths.output_dir=saves/eval/${unlearn_task_name}
```

### 🧭 Baseline

To reproduce the results in **Table 2** and **Figures 3-4** of our paper, please run the shell script `1B-FedAvg-TOFU-Split99.sh` inside the `scripts` folder.

![sample_result](./assets/sample_result.png)

<p align="center">Figure 2. Sample results of reproduction.
</p>


## 🔎 Roadmap

**Oblivionis** is still under development, and there is plenty of room for improvement. We will continue to update it and sincerely welcome suggestions and contributions to this open-source toolkit. If you encounter any issues or have questions, please feel free to open an issue in this repository.

Looking ahead, we plan to gradually include the following components:

- ⏳ Benchmark Datasets
- ⏳ Evaluation Metrics
- and more...

## 🤝 Acknowledgment

- This paper is inspired by [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM) and [OpenUnlearning](https://github.com/locuslab/open-unlearning), and also developed based on OpenUnlearning.
- Our implementation is built upon the [TOFU](https://github.com/locuslab/tofu) and [MUSE](https://github.com/swj0419/muse_bench) benchmarks.
- Our framework and paper take their name from [Lacus Oblivionis](https://en.wikipedia.org/wiki/Lacus_Oblivionis), a small lunar mare. Though, **to be honest**, it’s from [Ave Mujica](https://anime.bang-dream.com/avemujica/), a Japanese band that is part of [Bushiroad](https://bushiroad.com/)'s [BanG Dream!](https://en.bang-dream.com/) media franchise.
- To every **MyGO!!!!! (迷途之子!!!!!)**,

> **「僕は僕の味方でいようよ、まだまだ長い道の途中」（『栞』より）**

## 🔖 License

**Oblivionis** is licensed under the MIT License. Please see the [`LICENSE`](https://github.com/fyzhang1/Oblivionis/blob/main/LICENSE) file for details.

## 📄 Citation

If you find this work useful, please kindly consider citing our paper:

```bibtex
@misc{zhang2025oblivionislightweightlearningunlearning,
      title={Oblivionis: A Lightweight Learning and Unlearning Framework for Federated Large Language Models}, 
      author={Fuyao Zhang and Xinyu Yan and Tiantong Wu and Wenjie Li and Tianxiang Chen and Yang Cao and Ran Yan and Longtao Huang and Wei Yang Bryan Lim and Qiang Yang},
      year={2025},
      eprint={2508.08875},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.08875}, 
}
```
