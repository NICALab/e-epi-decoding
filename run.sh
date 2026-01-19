#!/bin/bash
set -euo pipefail
export RUN_BASH_NAME="$(basename "$0" .sh)"
export CUDA_VISIBLE_DEVICES=1

PYTHON=python
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${BASE_DIR}/main.py"
CSV_DIR="${BASE_DIR}/data_csv"

n_fft=256
hop_length=64
batch_size=64
lr=3e-4
num_epochs=999999999999
val_ratio=0.2
seed=42
early_stop_patience=50

time_drop_width=4
time_stripes_num=2
freq_drop_width=6
freq_stripes_num=2
train_ratio=0.7
layers=medium
hidden_size=128

model_type=Cnn14NerveNet_small

use_normalization=False
use_clipping=False
spec_aug=True
mixup=True
mixup_alpha=0.1

use_windowed_dataset=True
window_size=9       
window_stride=1

$PYTHON $SCRIPT \
      --csv_dir "$CSV_DIR" \
      --model_type "$model_type" \
      --lr "$lr" \
      --n_fft "$n_fft" \
      --hop_length "$hop_length" \
      --time_drop_width "$time_drop_width" \
      --time_stripes_num "$time_stripes_num" \
      --freq_drop_width "$freq_drop_width" \
      --freq_stripes_num "$freq_stripes_num" \
      --batch_size "$batch_size" \
      --num_epochs "$num_epochs" \
      --val_ratio "$val_ratio" \
      --train_ratio "$train_ratio" \
      --early_stop_patience "$early_stop_patience" \
      --num_layers "$layers" \
      --hidden_size "$hidden_size" \
      --use_normalization "$use_normalization" \
      --use_clipping "$use_clipping" \
      --spec_aug "$spec_aug" \
      --mixup "$mixup" \
      --mixup_alpha "$mixup_alpha" \
      --seed "$seed" \
      --use_windowed_dataset "$use_windowed_dataset" \
      --window_size "$window_size" \
      --window_stride "$window_stride" \
      --base_dir "$BASE_DIR"\
      --split_seed "$seed"

