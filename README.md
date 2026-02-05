# e-epi-decoding

Official code for:

**Anti-fibrotic Electronic Epineurium for High-fidelity Chronic Peripheral Nerve Interface in Freely Moving Animals**

This repository trains and evaluates a deep learning model to decode behavior classes from e-epi neural recordings.

Repo: https://github.com/NICALab/e-epi-decoding.git

---

## 1) System specifications tested (not hard requirements)

**Tested OS**
- Ubuntu 22.04.5 LTS

**Tested Python**
- Python 3.11.10

**Tested hardware**
- CPU supported (slower); GPU recommended
- Tested GPU: NVIDIA RTX 3090 (24 GB VRAM)
- Tested CPU: Intel(R) Xeon(R) Silver 4214R
- Tested RAM: 384 GB

**Tested PyTorch**
- torch 2.4.0+cu118 (CUDA 11.8 build)

---

## 2) Installation guide

Typical install time: **~10 minutes** on a standard workstation.

### 2.1 Create environment
```bash
conda create -n epi python=3.11.10 -y
conda activate epi
```

### 2.2 Install repository + Python dependencies
```bash
git clone https://github.com/NICALab/e-epi-decoding.git
cd e-epi-decoding
pip install -r requirements.txt
```

### 2.3 Install PyTorch (tested)
PyTorch is not installed via `requirements.txt` because it depends on CPU vs CUDA.

**GPU (tested: CUDA 11.8 wheels)**
```bash
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

**CPU-only (example)**
```bash
pip install torch torchvision torchaudio
```

---

## 3) Demo data placement (for editors/reviewers)

The dataset is provided separately (e.g., as part of the submission package). To run the demo:

1) From the repository root (BASE_DIR), create/verify the folder:
```bash
mkdir -p data_csv
```

2) Place the provided demo dataset CSV files into:
- `BASE_DIR/data_csv/`  (i.e., `./data_csv/` from the repo root)

---

## 4) Demo / reproduction run

### 4.1 Command
From the repository root:
```bash
bash run.sh
```

### 4.2 Expected runtime
- **~2 hours** on RTX 3090 (24 GB VRAM) using the default `run.sh` settings (batch size 64).
- CPU-only runs are supported but may take substantially longer (depends on CPU and dataset size).

> Note: `run.sh` sets a large `num_epochs`, but training stops via `early_stop_patience` once validation macro-F1 does not improve.

### 4.3 Expected outputs
Outputs are written under a timestamped run directory under:
- `checkpoints/nerve_behavior_decoding/main/`

Two key artifacts will be saved:
1) **Model checkpoint** (best validation macro-F1), e.g.:
   - `.../best_model_f1.pt`
2) **Test-set confusion matrix (SVG)**, e.g.:
   - `.../best_test_f1.svg`

---

## 5) Instructions for use (running on your own CSVs)

To run on your own recordings:
1) Convert each session to a `.csv` file.
2) Place all session CSVs in `BASE_DIR/data_csv/`.
3) Run:
```bash
bash run.sh
```

**Important:** The code expects CSV inputs and will create train/val/test splits and train the decoder automatically.

---

## 6) Algorithm pseudocode (paper-style)

### Algorithm 1: Training
```text
Algorithm 1  Training procedure for behavior decoding network

Input:
  Dtrain = {(xi, yi)}i=1..N              # labeled segments → class
  Dval   = {(xj, yj)}j=1..M              # validation set
  fθ(·)                                  # neural network (architecture abstracted)
  η, B, E                                # learning rate, batch size, max epochs
  P                                      # early-stopping patience
  ℒ(·,·)                                  # multi-class cross-entropy loss

Output:
  θ*                                     # parameters chosen by best validation macro-F1

1:  Initialize θ
2:  Initialize optimizer (e.g., Adam) with learning rate η
3:  bestF1 ← −∞ ; θ* ← θ ; wait ← 0
4:  for epoch = 1 to E do
5:      for each mini-batch {(x, y)} of size B from Dtrain do
6:          z ← fθ(x)                            # logits
7:          L ← ℒ(z, y)
8:          θ ← OptimizerStep(θ, ∇θ L)            # backprop + update
9:      end for
10:     F1 ← MacroF1( fθ(x), y ) on Dval
11:     if F1 > bestF1 then
12:         bestF1 ← F1 ; θ* ← θ ; wait ← 0
13:     else
14:         wait ← wait + 1
15:     end if
16:     if wait ≥ P then break end if
17: end for
18: return θ*
```

### Algorithm 2: Inference
```text
Algorithm 2  Inference procedure for unseen signal segments

Input:
  X = {xk}k=1..K                         # unseen segments
  fθ*(·)                                  # trained network
  π(·)                                    # class-index → label mapping

Output:
  Ŷ = {ŷk}k=1..K                          # predicted labels

1:  Set fθ* to inference mode
2:  for k = 1 to K do
3:      z ← fθ*(xk)                        # logits
4:      p ← Softmax(z)                     # probabilities
5:      ck ← argmax(p)                     # predicted class index
6:      ŷk ← π(ck)
7:  end for
8:  return Ŷ
```

---

## 7) License
This project is released under the **GNU General Public License v3.0 (GPL-3.0)**.
