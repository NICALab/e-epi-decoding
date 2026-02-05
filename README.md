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
