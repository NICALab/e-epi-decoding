import os
import re
import datetime
import argparse
import json
import random
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import f1_score

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial'] + mpl.rcParams['font.sans-serif']

def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
LABEL_MAP = {'walking': 0, 'climbing': 1, 'resting': 2, 'grooming': 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)

def generate_seed_based_split(seed, csv_dir):

    def _parse_week_from_name(fname: str) -> int:
        m = re.search('_(\\d+)[wW]_', fname)
        if m is None:
            m = re.search('(\\d+)[wW]', fname)
        if m is None:
            raise ValueError(f'Week pattern not found in filename: {fname}')
        return int(m.group(1))
    rng = np.random.RandomState(seed)
    csv_files = sorted([os.path.basename(f) for f in os.listdir(csv_dir) if f.endswith('.csv')])
    weeks_dict = {}
    for fname in csv_files:
        week = _parse_week_from_name(fname)
        if week not in weeks_dict:
            weeks_dict[week] = []
        weeks_dict[week].append(fname)
    split_dict = {'train': [], 'val': [], 'test': []}
    for week, filenames in sorted(weeks_dict.items()):
        num_sessions = len(filenames)
        if num_sessions < 3:
            split_dict['train'].extend(filenames)
        else:
            shuffled = filenames.copy()
            rng.shuffle(shuffled)
            split_dict['val'].append(shuffled[0])
            split_dict['test'].append(shuffled[1])
            split_dict['train'].extend(shuffled[2:])
    return split_dict

class WindowedNerveCSVDataset(Dataset):

    def __init__(self, csv_dir, window_size=9, stride=1, min_chunk_size=None, split_seed=None):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError(f'window_size must be odd to have a true center bin, got {window_size}')
        self.csv_dir = csv_dir
        self.window_size = window_size
        self.stride = stride
        self.min_chunk_size = min_chunk_size if min_chunk_size is not None else window_size
        if split_seed is None:
            split_seed = 0
        seed_split = generate_seed_based_split(split_seed, csv_dir)
        self.filename_to_split = {f: split for split, files in seed_split.items() for f in files}
        self.session_signals = {}
        self.session_labels = {}
        self.session_files = []
        self.session_weeks = []
        self.window_metadata = []
        csv_files = sorted([os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')])
        if len(csv_files) == 0:
            raise ValueError(f'No CSV files found in {csv_dir}')

        def _parse_week_from_name(fname: str) -> int:
            m = re.search('_(\\d+)[wW]_', fname)
            if m is None:
                m = re.search('(\\d+)[wW]', fname)
            if m is None:
                raise ValueError(f'Week pattern not found in filename: {fname}')
            return int(m.group(1))
        total_chunks = 0
        total_windows = 0
        split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
        label_counts = Counter()
        actual_session_id = 0
        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            week = _parse_week_from_name(fname)
            split = self.filename_to_split.get(fname, 'unknown')
            if split == 'unknown':
                continue
            df = pd.read_csv(csv_path)
            signals = df.iloc[:, :-1].values.astype(np.float32)
            labels_str = df.iloc[:, -1].astype(str).values
            self.session_signals[actual_session_id] = signals
            self.session_labels[actual_session_id] = labels_str
            self.session_files.append(fname)
            self.session_weeks.append(week)
            chunks = self._find_chunks(labels_str)
            total_chunks += len(chunks)
            for chunk_start, chunk_end, chunk_label in chunks:
                chunk_size = chunk_end - chunk_start + 1
                if chunk_size < self.min_chunk_size:
                    continue
                num_windows = (chunk_size - window_size) // stride + 1
                for w in range(num_windows):
                    window_start = chunk_start + w * stride
                    label_id = LABEL_MAP[chunk_label]
                    self.window_metadata.append((actual_session_id, window_start, label_id, split))
                    split_counts[split] += 1
                    label_counts[chunk_label] += 1
                    total_windows += 1
            actual_session_id += 1

    def _find_chunks(self, labels_str):
        chunks = []
        current_label = None
        current_start = None
        for i, label in enumerate(labels_str):
            if label in LABEL_MAP:
                if label == current_label:
                    pass
                else:
                    if current_label is not None:
                        chunks.append((current_start, i - 1, current_label))
                    current_label = label
                    current_start = i
            elif current_label is not None:
                chunks.append((current_start, i - 1, current_label))
                current_label = None
                current_start = None
        if current_label is not None:
            chunks.append((current_start, len(labels_str) - 1, current_label))
        return chunks

    def __len__(self):
        return len(self.window_metadata)

    def __getitem__(self, idx):
        session_id, window_start, label_id, split = self.window_metadata[idx]
        window_signals = self.session_signals[session_id][window_start:window_start + self.window_size]
        window_concat = window_signals.flatten()
        x = torch.from_numpy(window_concat.copy()).float().unsqueeze(0)
        y = torch.tensor(label_id, dtype=torch.long)
        return (x, y)

    def get_split(self, idx):
        return self.window_metadata[idx][3]

def split_windowed_dataset(dataset):
    train_idx = []
    val_idx = []
    test_idx = []
    for i, (_, _, _, split) in enumerate(dataset.window_metadata):
        if split == 'train':
            train_idx.append(i)
        elif split == 'val':
            val_idx.append(i)
        elif split == 'test':
            test_idx.append(i)
    return (train_idx, val_idx, test_idx)

@torch.no_grad()
def eval_sessions_stride1(model, windowed_dataset, session_indices, device, criterion, num_classes=NUM_CLASSES, split_name='val', batch_size=64):
    model.eval()
    window_size = windowed_dataset.window_size
    sessions_to_eval = set()
    for idx in session_indices:
        session_id, _, _, _ = windowed_dataset.window_metadata[idx]
        sessions_to_eval.add(session_id)
    all_preds = []
    all_trues = []
    all_losses = []
    for session_id in sorted(sessions_to_eval):
        session_signals = windowed_dataset.session_signals[session_id]
        session_labels_str = windowed_dataset.session_labels[session_id]
        num_bins = session_signals.shape[0]
        gt_labels = np.array([LABEL_MAP.get(l, -1) for l in session_labels_str], dtype=np.int64)
        all_windows = []
        all_center_bins = []
        for start in range(0, num_bins - window_size + 1):
            center_bin = start + window_size // 2
            window = session_signals[start:start + window_size]
            window_concat = window.flatten()
            all_windows.append(window_concat)
            all_center_bins.append(center_bin)
        if len(all_windows) == 0:
            continue
        for batch_start in range(0, len(all_windows), batch_size):
            batch_end = min(batch_start + batch_size, len(all_windows))
            batch_windows = all_windows[batch_start:batch_end]
            batch_center_bins = all_center_bins[batch_start:batch_end]
            batch_tensor = torch.stack([torch.from_numpy(w.copy()).float() for w in batch_windows])
            batch_tensor = batch_tensor.unsqueeze(1).to(device)
            logits = model(batch_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
            for i, center_bin in enumerate(batch_center_bins):
                gt_label = gt_labels[center_bin]
                if gt_label >= 0:
                    y = torch.tensor([gt_label], dtype=torch.long, device=device)
                    sample_logits = logits[i:i + 1]
                    loss = criterion(sample_logits, y)
                    all_preds.append(preds[i])
                    all_trues.append(gt_label)
                    all_losses.append(loss.item())
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    avg_acc = (all_preds == all_trues).mean() if len(all_preds) > 0 else 0.0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_trues, all_preds):
        conf_mat[t, p] += 1
    per_class_acc = {}
    for c in range(num_classes):
        class_total = (all_trues == c).sum()
        class_correct = ((all_trues == c) & (all_preds == c)).sum()
        per_class_acc[ID_TO_LABEL[c]] = class_correct / class_total if class_total > 0 else np.nan
    macro_f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_trues, all_preds, average=None, zero_division=0)
    f1_per_class_dict = {}
    for i in range(num_classes):
        f1_per_class_dict[ID_TO_LABEL[i]] = per_class_f1[i] if i < len(per_class_f1) else np.nan
    return (avg_loss, avg_acc, conf_mat, per_class_acc, all_trues, all_preds, np.array([]), macro_f1, f1_per_class_dict)

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x

class SpecAugmentation(nn.Module):

    def __init__(self, time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2):
        super().__init__()
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num

    def forward(self, x):
        if not self.training:
            return x
        B, C, F, T = x.shape
        for _ in range(self.freq_stripes_num):
            if self.freq_drop_width <= 0 or F <= 0:
                break
            w = torch.randint(low=1, high=self.freq_drop_width + 1, size=(1,)).item()
            if w >= F:
                continue
            f0 = torch.randint(low=0, high=F - w, size=(1,)).item()
            x[:, :, f0:f0 + w, :] = 0
        for _ in range(self.time_stripes_num):
            if self.time_drop_width <= 0 or T <= 0:
                break
            w = torch.randint(low=1, high=self.time_drop_width + 1, size=(1,)).item()
            if w >= T:
                continue
            t0 = torch.randint(low=0, high=T - w, size=(1,)).item()
            x[:, :, :, t0:t0 + w] = 0
        return x

class Cnn14NerveNet_small(nn.Module):

    def __init__(self, num_classes: int, n_fft: int=256, hop_length: int=128, sample_rate: int=24414, mel_bins: int=64, fmin: float=0.0, fmax: float | None=None, time_drop_width: int=4, time_stripes_num: int=1, freq_drop_width: int=6, freq_stripes_num: int=1, num_layers: str='small', last_hidden_size: int=128, spec_aug: bool=True, mixup: bool=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.num_layers = num_layers
        self.hidden_size = last_hidden_size
        self._spec_save_counter = 0
        self.spec_aug = spec_aug
        self.mixup = mixup
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(time_drop_width=time_drop_width, time_stripes_num=time_stripes_num, freq_drop_width=freq_drop_width, freq_stripes_num=freq_stripes_num)
        freq_bins = n_fft // 2 + 1
        self.bn0 = nn.BatchNorm2d(freq_bins)
        if self.num_layers == 'small':
            self.conv_block1 = ConvBlock(in_channels=1, out_channels=self.hidden_size // 2)
            self.conv_block2 = ConvBlock(in_channels=self.hidden_size // 2, out_channels=self.hidden_size)
            self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.fc_out = nn.Linear(self.hidden_size, num_classes, bias=True)
            self._init_weights()
        elif self.num_layers == 'medium':
            self.conv_block1 = ConvBlock(in_channels=1, out_channels=self.hidden_size // 8)
            self.conv_block2 = ConvBlock(in_channels=self.hidden_size // 8, out_channels=self.hidden_size // 4)
            self.conv_block3 = ConvBlock(in_channels=self.hidden_size // 4, out_channels=self.hidden_size // 2)
            self.conv_block4 = ConvBlock(in_channels=self.hidden_size // 2, out_channels=self.hidden_size)
            self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.fc_out = nn.Linear(self.hidden_size, num_classes, bias=True)
            self._init_weights()
        elif self.num_layers == 'large':
            self.conv_block1 = ConvBlock(in_channels=1, out_channels=self.hidden_size // 32)
            self.conv_block2 = ConvBlock(in_channels=self.hidden_size // 32, out_channels=self.hidden_size // 16)
            self.conv_block3 = ConvBlock(in_channels=self.hidden_size // 16, out_channels=self.hidden_size // 8)
            self.conv_block4 = ConvBlock(in_channels=self.hidden_size // 8, out_channels=self.hidden_size // 4)
            self.conv_block5 = ConvBlock(in_channels=self.hidden_size // 4, out_channels=self.hidden_size // 2)
            self.conv_block6 = ConvBlock(in_channels=self.hidden_size // 2, out_channels=self.hidden_size)
            self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.fc_out = nn.Linear(self.hidden_size, num_classes, bias=True)
            self._init_weights()

    def _init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_out)

    def _waveform_to_spec(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        assert C == 1, 'Input should be (B, 1, T)'
        waveform = x[:, 0, :]
        spec = self.spectrogram_extractor(waveform)
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
        spec = torch.log1p(spec)
        return spec

    def forward(self, x: torch.Tensor, mixup_index: torch.Tensor | None=None, mixup_lambda: float | None=None):
        x = self._waveform_to_spec(x)
        if self.training and self.spec_aug:
            x = self.spec_augmenter(x)
        if self.training and mixup_index is not None and (mixup_lambda is not None) and self.mixup:
            lam = float(mixup_lambda)
            x = lam * x + (1.0 - lam) * x[mixup_index, ...]
        if self.num_layers == 'small':
            x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
        elif self.num_layers == 'medium':
            x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
        elif self.num_layers == 'large':
            x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
            x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        logits = self.fc_out(x)
        return logits

def mixup_batch(y, alpha=0.2, device=None):
    if alpha <= 0:
        return (y, y, 1.0, None)
    if device is None:
        device = y.device
    batch_size = y.size(0)
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(batch_size, device=device)
    y_a = y
    y_b = y[index]
    return (y_a, y_b, lam, index)

def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.2):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if mixup_alpha > 0:
            y_a, y_b, lam, index = mixup_batch(y, alpha=mixup_alpha, device=device)
            logits = model(x, mixup_index=index, mixup_lambda=lam)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        else:
            logits = model(x)
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return (avg_loss, acc)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, num_classes=NUM_CLASSES, id_to_name=ID_TO_LABEL):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_logits = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    avg_loss = total_loss / total
    avg_acc = correct / total
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    logits_all = np.concatenate(all_logits, axis=0)
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1
    per_class_acc = {}
    for cls in range(num_classes):
        cls_name = id_to_name.get(cls, f'class_{cls}')
        mask = y_true == cls
        n_true = mask.sum()
        if n_true == 0:
            acc_cls = np.nan
        else:
            acc_cls = (y_pred[mask] == cls).mean()
        per_class_acc[cls_name] = acc_cls
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class_dict = {}
    for i, cls_name in enumerate([id_to_name.get(i, f'class_{i}') for i in range(num_classes)]):
        if i < len(per_class_f1):
            f1_per_class_dict[cls_name] = per_class_f1[i]
    return (avg_loss, avg_acc, conf_mat, per_class_acc, y_true, y_pred, logits_all, macro_f1, f1_per_class_dict)

def plot_confusion_matrix(conf_mat, class_names, acc=None, micro_f1=None, macro_f1=None, per_class_acc=None, title=None):
    conf_mat = conf_mat.astype(np.float32)
    row_sums = conf_mat.sum(axis=1, keepdims=True) + 1e-08
    conf_norm = conf_mat / row_sums
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.Blues
    im = ax.imshow(conf_norm, interpolation='nearest', aspect='auto', vmin=0.0, vmax=1.0, cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if title is not None:
        ax.set_title(title)
    else:
        title_parts = ['Confusion matrix (row-normalized)']
        if acc is not None:
            title_parts.append(f'acc={acc:.3f}')
        if micro_f1 is not None:
            title_parts.append(f'micro-F1={micro_f1:.3f}')
        if macro_f1 is not None:
            title_parts.append(f'macro-F1={macro_f1:.3f}')
        ax.set_title('\n'.join(title_parts))
    for i in range(conf_norm.shape[0]):
        for j in range(conf_norm.shape[1]):
            val = conf_norm[i, j]
            text_color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)
    fig.tight_layout()
    return fig

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False).')


def _sanitize(s: str, max_len: int=80) -> str:
    s = str(s)
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    s = s.strip('._-')
    if not s:
        s = 'run'
    return s[:max_len]

def save_confmat_svg(conf_mat, class_names, out_path, title=None, acc=None, macro_f1=None, per_class_acc=None):
    assert conf_mat.shape[0] == conf_mat.shape[1] == len(class_names)
    if per_class_acc is None:
        per_class_acc = {}
        row_sums = conf_mat.sum(axis=1)
        for i, name in enumerate(class_names):
            if row_sums[i] > 0:
                per_class_acc[name] = conf_mat[i, i] / row_sums[i]
            else:
                per_class_acc[name] = np.nan
    if acc is None:
        total = conf_mat.sum()
        correct = conf_mat.trace()
        acc = correct / total if total > 0 else 0.0
    micro_f1 = acc
    fig = plot_confusion_matrix(conf_mat, class_names, acc=acc, micro_f1=micro_f1, macro_f1=macro_f1, per_class_acc=per_class_acc)
    if title is not None:
        ax = fig.axes[0]
        ax.set_title(title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ax_ in fig.axes:
        ax_.patch.set_clip_on(False)
    for artist in fig.findobj():
        if hasattr(artist, 'set_clip_on'):
            artist.set_clip_on(False)
        if hasattr(artist, 'set_clip_path'):
            artist.set_clip_path(None)
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=None, help='Seed for generating train/val/test split. Defaults to --seed when not provided.')
    parser.add_argument('--model_type', type=str, default='Cnn14NerveNet_small', choices=['MobileNetV1', 'Cnn14NerveNet', 'Cnn6', 'WavegramLogmelCnn14NerveNet', 'ResNet38NerveNet', 'Cnn14NerveNet_small'])
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--hop_length', type=int, default=128)
    parser.add_argument('--time_drop_width', type=int, default=4)
    parser.add_argument('--time_stripes_num', type=int, default=1)
    parser.add_argument('--freq_drop_width', type=int, default=6)
    parser.add_argument('--freq_stripes_num', type=int, default=1)
    parser.add_argument('--num_layers', type=str, default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--spec_aug', type=str2bool, default=True)
    parser.add_argument('--mixup', type=str2bool, default=True)
    parser.add_argument('--use_normalization', type=str2bool, default=True)
    parser.add_argument('--use_clipping', type=str2bool, default=True)
    parser.add_argument('--early_stop_patience', type=int, default=100)
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint if available')
    parser.add_argument('--base_dir', type=str, default='./', help='Base directory for logs and checkpoints')
    parser.add_argument('--use_windowed_dataset', type=str2bool, default=False, help='Use WindowedNerveCSVDataset with sliding windows and concatenated bins')
    parser.add_argument('--window_size', type=int, default=9, help='Window size for windowed dataset (MUST BE ODD). E.g., 9 means 9 bins concatenated.')
    parser.add_argument('--window_stride', type=int, default=1, help='Stride for sliding window in training')
    args = parser.parse_args()

    if args.use_windowed_dataset and args.window_size % 2 == 0:
        raise ValueError(f'--window_size must be odd to have a true center bin, got {args.window_size}')

    CSV_DIR = args.csv_dir
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LR = args.lr
    N_FFT = args.n_fft
    HOP_LENGTH = args.hop_length
    EARLY_STOP_PATIENCE = args.early_stop_patience
    RESUME_TRAINING = args.resume
    time_drop_width = args.time_drop_width
    time_stripes_num = args.time_stripes_num
    freq_drop_width = args.freq_drop_width
    freq_stripes_num = args.freq_stripes_num
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    spec_aug = args.spec_aug
    mixup = args.mixup
    mixup_alpha = args.mixup_alpha
    window_size = args.window_size
    window_stride = args.window_stride
    use_windowed = args.use_windowed_dataset
    SEED = args.seed
    SPLIT_SEED = args.split_seed if args.split_seed is not None else SEED
    base_dir = args.base_dir

    seed_everything(SEED)

    script_name = os.path.basename(__file__)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bash_name = os.environ.get('RUN_BASH_NAME', 'direct')
    bash_name = _sanitize(bash_name)
    split_seed_tag = f'splitseed{SPLIT_SEED}'

    ckpt_dir = os.path.join(base_dir, f'checkpoints/nerve_behavior_decoding/{script_name[:-3]}', f'main_{timestamp}_stride{window_stride}_window{window_size}_bash{bash_name}_{split_seed_tag}')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_f1_ckpt_path = os.path.join(ckpt_dir, 'best_model_f1.pt')

    if not use_windowed:
        raise ValueError('This script expects --use_windowed_dataset True (as in run.sh).')

    print("Loading dataset...")
    dataset = WindowedNerveCSVDataset(csv_dir=CSV_DIR, window_size=window_size, stride=window_stride, split_seed=SPLIT_SEED)
    train_idx, val_idx, test_idx = split_windowed_dataset(dataset)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    device = torch.device(DEVICE)

    if args.model_type != 'Cnn14NerveNet_small':
        raise ValueError(f'Unsupported model_type: {args.model_type}')

    model = Cnn14NerveNet_small(num_classes=NUM_CLASSES, n_fft=N_FFT, hop_length=HOP_LENGTH, sample_rate=24414, mel_bins=64, fmin=0.0, fmax=12207.0, time_drop_width=time_drop_width, time_stripes_num=time_stripes_num, freq_drop_width=freq_drop_width, freq_stripes_num=freq_stripes_num, num_layers=num_layers, last_hidden_size=hidden_size, spec_aug=spec_aug, mixup=mixup).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if RESUME_TRAINING and os.path.exists(best_f1_ckpt_path):
        ckpt = torch.load(best_f1_ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    train_labels = [dataset.window_metadata[i][2] for i in train_idx]
    label_counts = Counter([int(l) for l in train_labels])
    class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
    sample_weights = [class_weights[int(l)] for l in train_labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    g_sampler = torch.Generator()
    g_sampler.manual_seed(SEED)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True, generator=g_sampler)

    NUM_WORKERS = 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker if NUM_WORKERS > 0 else None, generator=torch.Generator().manual_seed(SEED), persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker if NUM_WORKERS > 0 else None, generator=torch.Generator().manual_seed(SEED), persistent_workers=NUM_WORKERS > 0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker if NUM_WORKERS > 0 else None, generator=torch.Generator().manual_seed(SEED), persistent_workers=NUM_WORKERS > 0)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_opt_state = None
    no_improve_count = 0
    MIN_DELTA = 0.0

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc='Training...'):
        train_one_epoch(model, train_loader, optimizer, criterion, device, mixup_alpha=mixup_alpha)
        val_loss, val_acc, conf_mat, per_class_acc, y_true, y_pred, logits_all, macro_f1, f1_per_class = eval_sessions_stride1(model, dataset, val_idx, device, criterion, num_classes=NUM_CLASSES, split_name='val', batch_size=BATCH_SIZE)

        if macro_f1 > best_val_f1 + MIN_DELTA:
            print(f"  [*] New best-F1 model saved at epoch {epoch:02d}")
            best_val_f1 = float(macro_f1)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt_state = optimizer.state_dict()
            no_improve_count = 0

            # Save checkpoint
            save_dict_f1 = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                'seed': SEED,
                'split_seed': SPLIT_SEED,
            }
            torch.save(save_dict_f1, best_f1_ckpt_path)
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"[Early Stop] Macro F1 has not improved for "
                  f"{EARLY_STOP_PATIENCE} epochs. Stopping at epoch {epoch:02d}.")
            save_dict_f1 = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                'seed': SEED,
                'split_seed': SPLIT_SEED,
            }
            torch.save(save_dict_f1, best_f1_ckpt_path)
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_opt_state = optimizer.state_dict()
        best_epoch = 0
        best_val_f1 = -1.0

    model.load_state_dict(best_state)

    test_loss, test_acc, test_conf_mat, test_per_class_acc, test_y_true, test_y_pred, test_logits_all, test_macro_f1, test_f1_per_class = eval_sessions_stride1(model, dataset, test_idx, device, criterion, num_classes=NUM_CLASSES, split_name='test', batch_size=BATCH_SIZE)

    class_names = [ID_TO_LABEL[i] for i in range(NUM_CLASSES)]
    out_test = os.path.join(ckpt_dir, 'best_test_f1.svg')
    save_confmat_svg(test_conf_mat, class_names, out_test, title=f'epoch={best_epoch}, acc={test_acc:.3f}, macro-F1={test_macro_f1:.3f}', acc=test_acc, macro_f1=test_macro_f1, per_class_acc=test_per_class_acc)

    save_dict_f1 = {
        'epoch': best_epoch,
        'model_state_dict': best_state,
        'optimizer_state_dict': best_opt_state,
        'best_val_f1': best_val_f1,
        'seed': SEED,
        'split_seed': SPLIT_SEED,
    }
    torch.save(save_dict_f1, best_f1_ckpt_path)
