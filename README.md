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

## Expected data directory structure

Download the CSV files from the data provided by the paper and place them under `e-epi-decoding/data_csv/`.

## Install required packages

```bash
git clone https://github.com/NICALab/e-epi-decoding.git
cd e-epi-decoding
pip install -r requirements.txt
```

## Reproduce results from paper

```bash
cd e-epi-decoding
bash run.sh
```

This will train the model and write checkpoints + the test confusion matrix figure under:

- `e-epi-decoding/checkpoints/nerve_behavior_decoding/main/main_<timestamp>_stride<...>_window<...>_bash<...>_splitseed<...>/`
