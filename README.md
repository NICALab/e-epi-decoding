# e-epi-decoding

Official code for:

**Anti-fibrotic Electronic Epineurium for High-fidelity Chronic Peripheral Nerve Interface in Freely Moving Animals**

This repository trains and evaluates a deep learning model to decode behavior classes from e-epi neural recordings.

Repository link: https://github.com/NICALab/e-epi-decoding.git


## 1) System specifications used to verify software

**Operating System (OS)**
- Ubuntu 22.04.5 LTS

**Hardware**
- CPU supported (slower); GPU recommended
- GPU: NVIDIA RTX 3090 (24 GB VRAM)
- CPU: Intel(R) Xeon(R) Silver 4214R
- RAM: 384 GB

**Software dependencies** (also found in requirements.txt)
- Python 3.11.10
- torch 2.4.0+cu118 (CUDA 11.8 build)
- torchaudio 2.4.0+cu118
- torchvision 0.19.0+cu118
- torchlibrosa 0.1.0
- numpy 1.26.4
- pandas 1.5.3
- tqdm 4.66.5
- scikit-learn 1.5.2
- matplotlib =3.9.2


## 2) Installation guide

Typical install time: **~10 minutes** on a standard workstation.

### 2.1 Create conda environment
```bash
conda create -n epi python=3.11.10 -y
conda activate epi
```

### 2.2 Install PyTorch
PyTorch is not installed via `requirements.txt` because it depends on computational hardware: CPU vs GPU (CUDA).

**GPU (verified: CUDA 11.8 wheels)**
```bash
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

**CPU-only (example)**
```bash
pip install torch torchvision torchaudio
```

### 2.3 Download code repository + Install dependencies
```bash
git clone https://github.com/NICALab/e-epi-decoding.git
cd e-epi-decoding
pip install -r requirements.txt
```

## 3) Demo data placement

The dataset is provided separately at the data repository link. To run the demo with the demo data from study:

1) From the repository root (BASE_DIR), create/verify the folder:
```bash
mkdir -p data_csv
```

2) Place the provided demo dataset CSV files into:
- `BASE_DIR/data_csv/`  (i.e., `./data_csv/` from the repo root)


## 4) Run training and reproduce results

### 4.1 Command
After placing the data in the correct directory, run the following command from the repository root (i.e. after `cd e-epi-decoding`):
```bash
bash run.sh
```
Typical run time: **~2 hours** on a single RTX 3090 (24 GB VRAM) using the default `run.sh` settings (batch size 64). CPU-only runs are supported but may take substantially longer.

> Note: The default settings in `run.sh` sets a large `num_epochs`, but training stops via `early_stop_patience` once validation F1 score does not improve.

### 4.2 Expected outputs
Outputs are written under a timestamped run directory under:
- `BASE_DIR/checkpoints/nerve_behavior_decoding/main/`

Two files will be saved:
1) **Model checkpoint** (best validation F1 score), e.g.:
   - `.../best_model_f1.pt`
2) **Test-set confusion matrix (SVG) and quantitative decoding results**, e.g.:
   - `.../best_test_f1.svg`


## 6) Pseudocode

### Training
```text
1. Training procedure for behavior decoding network

Symbols:
  Dtrain = {(xi, yi)}i=1..N              # labeled segments → class
  θ                                      # trainable model parameters
  fθ(·)                                  # neural network (architecture abstracted)
  η, B                                   # learning rate, batch size
  ℒ(·,·)                                 # multi-class cross-entropy loss


1:  Initialize θ
2:  Initialize optimizer with learning rate η
3:  for each epoch do
4:      for each mini-batch {(x, y)} of size B from Dtrain do
5:          z ← fθ(x)                     # logits
6:          L ← ℒ(z, y)
7:          θ ← OptimizerStep(θ, ∇θ L)    # backprop + update
8:      end for
9:  end for
```
### Inference
```text
2.  Inference procedure for unseen trial data

Symbols:
  X = {xk}k=1..K                          # unseen data
  fθ*(·)                                  # trained network

1:  Set fθ* to inference mode
2:  for k = 1 to K do
3:      z ← fθ*(xk)                       # logits
4:      p ← Softmax(z)                    # probabilities
5:      ŷk ← argmax(p)                    # predicted class index
6:  end for
```

---

## 7) License
This project is released under the **GNU General Public License v3.0 (GPL-3.0)**.
