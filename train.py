import os
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

# =========================================================
# 1) CONFIG
# =========================================================
@dataclass
class Config:
    train_dir: str = "/kaggle/input/datasets/anhcngnguyn/deepfake-audio-train/train_data/train_set"
    val_dir:   str = "/kaggle/input/datasets/anhcngnguyn/deepfake-audio-train/train_data/val_set"
    
    model_name: str = "microsoft/wavlm-base"
    
    target_sr: int = 16000
    clip_seconds: int = 4
    
    batch_size: int = 8
    num_epochs: int = 8
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    
    num_classes: int = 2
    dropout: float = 0.3
    head_hidden_dim: int = 256
    
    freeze_backbone: bool = False
    unfreeze_last_n_layers: int = 4
    
    use_class_weight: bool = True
    save_path: str = "/kaggle/working/best_model_wavlm.pt"

    # 🔥 Early stopping
    early_stopping_patience: int = 3

CFG = Config()

# =========================================================
# 2) DEVICE + SEED
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"Using device: {device}")

# =========================================================
# 3) DATA
# =========================================================
def collect_labeled_files(data_dir: str) -> List[Tuple[str, int]]:
    real_files = glob.glob(os.path.join(data_dir, "real", "*.wav"))
    fake_files = glob.glob(os.path.join(data_dir, "fake", "*.wav"))
    samples = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
    random.shuffle(samples)
    return samples

def preprocess_audio(path: str, target_sr: int, clip_seconds: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.squeeze(0)
    wav = wav / (wav.abs().max() + 1e-9)

    target_len = target_sr * clip_seconds
    if wav.numel() >= target_len:
        wav = wav[:target_len]
    else:
        wav = F.pad(wav, (0, target_len - wav.numel()))

    return wav

class DeepfakeTrainDataset(Dataset):
    def __init__(self, samples, target_sr, clip_seconds):
        self.samples = samples
        self.target_sr = target_sr
        self.clip_seconds = clip_seconds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = preprocess_audio(path, self.target_sr, self.clip_seconds)
        return {
            "waveform": wav,
            "label": torch.tensor(label, dtype=torch.long),
        }

feature_extractor = AutoFeatureExtractor.from_pretrained(CFG.model_name)

def train_collate_fn(batch):
    waveforms = [item["waveform"].numpy() for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    inputs = feature_extractor(
        waveforms,
        sampling_rate=CFG.target_sr,
        return_tensors="pt",
        padding=True,
    )

    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs.get("attention_mask"),
        "labels": labels,
    }

# =========================================================
# 4) MODEL
# =========================================================
class MeanStdPooling(nn.Module):
    def forward(self, hs, attention_mask=None):
        if attention_mask is None:
            return torch.cat([hs.mean(1), hs.std(1)], dim=1)

        mask = attention_mask.unsqueeze(-1).float()
        lengths = mask.sum(dim=1).clamp(min=1e-6)

        mean = (hs * mask).sum(dim=1) / lengths
        var = (((hs - mean.unsqueeze(1)) * mask) ** 2).sum(dim=1) / lengths
        std = torch.sqrt(var + 1e-6)

        return torch.cat([mean, std], dim=1)

class AudioDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CFG.model_name)
        hidden = self.backbone.config.hidden_size

        self.pool = MeanStdPooling()

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, CFG.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.head_hidden_dim, CFG.num_classes),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hs = outputs.last_hidden_state

        feat_mask = None
        if attention_mask is not None:
            feat_mask = self.backbone._get_feature_vector_attention_mask(
                hs.shape[1], attention_mask
            )

        emb = self.pool(hs, feat_mask)
        logits = self.classifier(emb)
        return logits

def configure_trainable_layers(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

    if not CFG.freeze_backbone:
        if CFG.unfreeze_last_n_layers == 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
        elif hasattr(model.backbone, "encoder"):
            layers = model.backbone.encoder.layers
            for layer in layers[-CFG.unfreeze_last_n_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True

# =========================================================
# 5) METRICS
# =========================================================
def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "specificity": specificity,
    }

# =========================================================
# TRAIN LOOP
# =========================================================
def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()

    total_loss = 0
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for batch in loader:
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device)
        mask = batch["attention_mask"]
        if mask is not None:
            mask = mask.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(x, mask)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true_all.extend(y.cpu().tolist())
        y_pred_all.extend(preds.cpu().tolist())
        y_prob_all.extend(probs.cpu().tolist())

    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics

# =========================================================
# MAIN
# =========================================================
def main():
    train_samples = collect_labeled_files(CFG.train_dir)
    val_samples   = collect_labeled_files(CFG.val_dir)

    train_loader = DataLoader(
        DeepfakeTrainDataset(train_samples, CFG.target_sr, CFG.clip_seconds),
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=train_collate_fn
    )

    val_loader = DataLoader(
        DeepfakeTrainDataset(val_samples, CFG.target_sr, CFG.clip_seconds),
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=train_collate_fn
    )

    model = AudioDeepfakeDetector().to(device)
    configure_trainable_layers(model)

    # class weight
    if CFG.use_class_weight:
        counts = torch.tensor([
            sum(1 for _, y in train_samples if y == 0),
            sum(1 for _, y in train_samples if y == 1),
        ], dtype=torch.float32)

        weights = counts.sum() / (len(counts) * counts)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr_head)

    # 🔥 Early stopping
    best_f1 = -1
    patience = CFG.early_stopping_patience
    counter = 0

    for epoch in range(CFG.num_epochs):
        train_m = run_epoch(model, train_loader, optimizer, criterion, True)
        val_m   = run_epoch(model, val_loader, optimizer, criterion, False)

        print(f"\nEpoch {epoch+1}")
        print(f"Train F1: {train_m['f1']:.4f}")
        print(f"Val   F1: {val_m['f1']:.4f} | AUC: {val_m['auc']:.4f}")

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            counter = 0
            torch.save(model.state_dict(), CFG.save_path)
            print("✅ Save best model")
        else:
            counter += 1
            print(f"⏳ EarlyStopping: {counter}/{patience}")

            if counter >= patience:
                print("🛑 Early stopping!")
                break

    print("Training done!")

if __name__ == "__main__":
    main()