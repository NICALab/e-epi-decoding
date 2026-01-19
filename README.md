# e-epi-decoding

This repository will provide the official code, pretrained models, and datasets for the paper:

"Anti-fibrotic Electronic Epineurium for High-fidelity Chronic Peripheral Nerve Interface in Freely Moving Animals"

In this work, we demonstrate behavior decoding using signals recorded from our novel electronic epineurium (e-epi) interface.

## Directory setup

From the repository root:

- `e-epi-decoding/`
  - `main.py`
  - `run.sh`
  - `requirements.txt`
  - `data_csv/`  *(create this and put your CSVs here)*

## Expected data format

Place one or more `.csv` files under `e-epi-decoding/data_csv/`.

- **CSV columns**:
  - All columns **except the last** are numeric signal features (float-compatible).
  - The **last column** is the label string for each time bin (one of): `walking`, `climbing`, `resting`, `grooming`.

## Install required packages

From `e-epi-decoding/`:

```bash
pip install -r requirements.txt
```

## Reproduce results (run training/eval)

From `e-epi-decoding/`:

```bash
bash run.sh
```

This will train the model and write checkpoints + the test confusion matrix figure under:

- `e-epi-decoding/checkpoints/nerve_behavior_decoding/main/main_<timestamp>_stride<...>_window<...>_bash<...>_splitseed<...>/`
