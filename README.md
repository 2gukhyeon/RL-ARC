This repository contains the official code for the paper:

> **RL-ARC: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty**  
> Gukhyeon Lee and SangKeun Lee \
> \[[Paper Link](https://arxiv.org/abs/2610.11352)\]


We are currently refining and organizing the codebase, and will release the cleaned version soon.

---

## News
🎉 RL-ARC has been selected for ORAL presentation at AACL-IJCNLP 2026!

🎉 RL-ARC has been accepted to AACL-IJCNLP 2026 (Main)!

---

## 📄 Citation

```bibtex
@inproceedings{lee-lee-2026-rl,
    title = "{RL}-{ARC}: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty",
    author = "Lee, Gukhyeon  and
      Lee, SangKeun",
    editor = "Zhang, Yue  and
      Xu, Ruifeng  and
      Purwarianti, Ayu  and
      Rong, Tong  and
      Suzuki, Jun",
    booktitle = "Proceedings of the 15th International Joint Conference on Natural Language Processing and the 5th Conference of the {A}sia-Pacific Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2026",
    address = "Hengqin, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.aacl-long.236/",
    doi = "10.18653/2026.aacl-long.236",
    pages = "5284--5310",
    ISBN = "979-8-89176-439-2",
    abstract = "Language models (LMs) are commonly trained with Reinforcement Learning with Verifiable Rewards (RLVR) to enhance their reasoning capabilities. However, since RLVR does not explicitly account for calibration during training, it can lead to severe calibration degradation, including overconfidence. Recent calibration-aware training methods for LMs, which incorporate objectives for uncertainty estimation into training, improve calibration but still exhibit overconfidence under distribution shift, while sacrificing reasoning performance. To this end, we propose RL-ARC, a calibration-aware training framework that jointly leverages reasoning confidence and answer confidence. Specifically, RL-ARC leverages reasoning confidence as an auxiliary signal for calibrating answer confidence, applying it as reasoning-guided regularization for correct cases and as an overconfidence penalty for incorrect cases. Comprehensive results across ID and OOD settings show that, beyond improving calibration, RL-ARC enables reasoning models to adaptively estimate confidence based on the given question without substantially sacrificing reasoning performance, thereby highlighting the importance of reasoning confidence for training reliable reasoning models."
}
```

```bibtex
@misc{lee2026rlarccalibratinglargereasoning,
      title={RL-ARC: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty}, 
      author={Gukhyeon Lee and SangKeun Lee},
      year={2026},
      eprint={2610.11352},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.11352}, 
}
```


---

## 🛠 Installation

### Environment Setup

#### 1. cloud GPU
  - CUDA version >= 12.1
  - torch version = 2.5.1
  - python version = 3.10
  - hard disk >= 160 GB
  
```bash
git clone https://github.com/2gukhyeon/RLAR_project.git
your git key!
cd RLAR_tentative 
pip install -r requirement.txt
```
#### 2. xx.40 server (ours)
  - CUDA version >= 12.1
  
  ```bash
  git clone https://github.com/2gukhyeon/RLAR_project.git
  conda create -n calibration python=3.10
  conda activate calibration
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -U pip setuptools wheel
  pip install ninja packaging
  conda install -c nvidia cuda-toolkit=12.1
  pip install -r requirement.txt (not including flash-attention)
  pip install flash-attn --no-build-isolation
  ```  

### TRL Installation 
```bash
cd ../
git clone https://github.com/huggingface/trl.git
cd trl/
git checkout 69ad852e5654a77f1695eb4c608906fe0c7e8624
pip install -e .
``` 

### Login to wandb 

```bash
wandb login
your wandb api key!
```

### Login to Huggingface

```bash
huggingface-cli login
your huggingface key!
Y
```

---
## 🚀 Training

To run post-training (e.g., RLAR, RLVR, RLCR, etc.) on hotpot:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file deepspeed.yaml rl_runner.py --config configs/Qwen-7B/hotpot/RLAR.yaml
```

To run post-training (e.g., RLAR, RLVR, RLCR, etc.) on math:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file deepspeed.yaml rl_runner.py --config configs/Qwen-7B/math/RLAR.yaml
```

## 📊 Inference

To run inference with our trained model on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 inference_example.py
```

### 🧪 Evaluation

Run evaluation on a dataset using a config:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py --config eval_configs/Hotpot-models/trivia.json
```

For a full eval suite on a single GPU (We already provide the outputs/results from this):

```bash
bash eval_runs.sh
```
