import os
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


# =========================================================
# 1) CONFIG
# =========================================================

@dataclass
class Config:
    train_dir: str = "data/train_data_v1"

    # Backbone: đổi giữa WavLM và wav2vec2
    model_name: str = "microsoft/wavlm-base"
    # model_name: str = "facebook/wav2vec2-base"

    target_sr: int = 16000
    clip_seconds: int = 4

    batch_size: int = 4
    num_epochs: int = 10
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0

    num_classes: int = 2
    dropout: float = 0.3
    head_hidden_dim: int = 256

    # pooling có thể đổi để làm thí nghiệm
    pooling: str = "mean_std"   # "mean" | "mean_std" | "attention"

    # chiến lược fine-tune
    freeze_backbone: bool = False
    unfreeze_last_n_layers: int = 4  # 0 = full fine-tune

    use_class_weight: bool = True
    save_path: str = "best_model.pt"


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


# =========================================================
# 3) DATA
# =========================================================

def collect_labeled_files(train_dir: str) -> List[Tuple[str, int]]:
    real_files = glob.glob(os.path.join(train_dir, "real", "*.wav"))
    fake_files = glob.glob(os.path.join(train_dir, "fake", "*.wav"))

    samples = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
    random.shuffle(samples)
    return samples


def preprocess_audio(path: str, target_sr: int, clip_seconds: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # [C, T]

    # multi-channel -> mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # [1, T] -> [T]
    wav = wav.squeeze(0)

    # normalize amplitude
    wav = wav / (wav.abs().max() + 1e-9)

    target_len = target_sr * clip_seconds

    # crop / pad
    if wav.numel() >= target_len:
        wav = wav[:target_len]
    else:
        wav = F.pad(wav, (0, target_len - wav.numel()))

    return wav


class DeepfakeTrainDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], target_sr: int, clip_seconds: int):
        self.samples = samples
        self.target_sr = target_sr
        self.clip_seconds = clip_seconds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        wav = preprocess_audio(path, self.target_sr, self.clip_seconds)
        return {
            "path": path,
            "waveform": wav,
            "label": torch.tensor(label, dtype=torch.long),
        }


feature_extractor = AutoFeatureExtractor.from_pretrained(CFG.model_name)


def train_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
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
        "paths": [item["path"] for item in batch],
    }


# =========================================================
# 4) POOLING
# =========================================================

class MeanPooling(nn.Module):
    def forward(self, hs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            return hs.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()
        summed = (hs * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


class MeanStdPooling(nn.Module):
    def forward(self, hs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            mean = hs.mean(dim=1)
            std = hs.std(dim=1)
            return torch.cat([mean, std], dim=1)

        mask = attention_mask.unsqueeze(-1).float()
        lengths = mask.sum(dim=1).clamp(min=1e-6)

        mean = (hs * mask).sum(dim=1) / lengths
        var = (((hs - mean.unsqueeze(1)) * mask) ** 2).sum(dim=1) / lengths
        std = torch.sqrt(var + 1e-6)

        return torch.cat([mean, std], dim=1)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.attn(hs).squeeze(-1)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (hs * weights).sum(dim=1)
        return pooled


# =========================================================
# 5) MODEL
# =========================================================

class AudioDeepfakeDetector(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        dropout: float = 0.3,
        head_hidden_dim: int = 256,
        pooling: str = "mean_std",
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        if pooling == "mean":
            self.pool = MeanPooling()
            pooled_dim = hidden_size
        elif pooling == "mean_std":
            self.pool = MeanStdPooling()
            pooled_dim = hidden_size * 2
        elif pooling == "attention":
            self.pool = AttentionPooling(hidden_size)
            pooled_dim = hidden_size
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hs = outputs.last_hidden_state # [B, T_feat, H]

        feature_attention_mask = None
        if attention_mask is not None:
            feature_attention_mask = self.backbone._get_feature_vector_attention_mask(
                hs.shape[1], attention_mask
            )

        emb = self.pool(hs, feature_attention_mask)
        logits = self.classifier(emb)

        return logits, emb


def configure_trainable_layers(model: AudioDeepfakeDetector, freeze_backbone: bool, unfreeze_last_n_layers: int):
    for p in model.backbone.parameters():
        p.requires_grad = False

    if not freeze_backbone:
        if unfreeze_last_n_layers == 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
        else:
            if hasattr(model.backbone, "encoder") and hasattr(model.backbone.encoder, "layers"):
                layers = model.backbone.encoder.layers
                for layer in layers[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

                if hasattr(model.backbone, "feature_projection"):
                    for p in model.backbone.feature_projection.parameters():
                        p.requires_grad = True
            else:
                for p in model.backbone.parameters():
                    p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True

    for p in model.pool.parameters():
        p.requires_grad = True


# =========================================================
# 6) METRICS
# =========================================================

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for batch in loader:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        logits, _ = model(input_values=input_values, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true_all.extend(labels.detach().cpu().tolist())
        y_pred_all.extend(preds.detach().cpu().tolist())
        y_prob_all.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for batch in loader:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        logits, _ = model(input_values=input_values, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true_all.extend(labels.detach().cpu().tolist())
        y_pred_all.extend(preds.detach().cpu().tolist())
        y_prob_all.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    metrics["loss"] = avg_loss
    return metrics


# =========================================================
# 7) MAIN TRAIN
# =========================================================

def main():
    print(f"Using device: {device}")
    print(f"Backbone: {CFG.model_name}")
    print(f"Pooling: {CFG.pooling}")

    samples = collect_labeled_files(CFG.train_dir)

    if len(samples) == 0:
        raise ValueError(f"Không tìm thấy file wav trong {CFG.train_dir}/real hoặc {CFG.train_dir}/fake")

    num_real = sum(1 for _, y in samples if y == 0)
    num_fake = sum(1 for _, y in samples if y == 1)

    print(f"Total train samples: {len(samples)}")
    print(f"Real: {num_real}")
    print(f"Fake: {num_fake}")

    full_dataset = DeepfakeTrainDataset(samples, CFG.target_sr, CFG.clip_seconds)

    val_size = int(len(full_dataset) * CFG.val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CFG.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=train_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=train_collate_fn,
    )

    model = AudioDeepfakeDetector(
        model_name=CFG.model_name,
        num_classes=CFG.num_classes,
        dropout=CFG.dropout,
        head_hidden_dim=CFG.head_hidden_dim,
        pooling=CFG.pooling,
    ).to(device)

    configure_trainable_layers(
        model,
        freeze_backbone=CFG.freeze_backbone,
        unfreeze_last_n_layers=CFG.unfreeze_last_n_layers,
    )

    if CFG.use_class_weight:
        class_counts = torch.tensor([num_real, num_fake], dtype=torch.float32)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name.startswith("backbone"):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CFG.lr_backbone},
            {"params": head_params, "lr": CFG.lr_head},
        ],
        weight_decay=CFG.weight_decay,
    )

    best_f1 = -1.0

    for epoch in range(1, CFG.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = validate_one_epoch(model, val_loader, criterion)

        print(f"\nEpoch {epoch}/{CFG.num_epochs}")
        print(
            f"Train | loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['acc']:.4f} "
            f"f1={train_metrics['f1']:.4f} "
            f"auc={train_metrics['auc']:.4f}"
        )
        print(
            f"Val   | loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), CFG.save_path)
            print(f"Saved best model to {CFG.save_path}")

    print(f"\nTraining xong. Best model đã lưu ở: {CFG.save_path}")


if __name__ == "__main__":
    main()