This repository contains the tentative code for the NeurIPS 2026 project.

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
