# facebook/hubert-base-ls960

import os
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# =========================================================
# 1) CONFIG
# =========================================================
@dataclass
class Config:
    # =====================================================
    # CHỌN DATASET Ở ĐÂY
    # =====================================================
    # "scene"  -> train trên SceneFake
    # "for"    -> train trên FoR
    # "merged" -> train trên Merged
    #
    # Gợi ý chạy thực nghiệm:
    # 1. dataset_mode = "scene"
    # 2. dataset_mode = "for"
    # 3. dataset_mode = "merged"
    # =====================================================
    dataset_mode: str = "merged"

    # Kaggle dataset root
    data_root: str = "/kaggle/input/datasets/anhcngnguyn/deepfake-audio-dataset"

    # Tên thư mục dataset trong data_root
    for_name: str = "FoR"
    scene_name: str = "SceneFake"
    merged_name: str = "Merged"

    # =====================================================
    # CHỌN MODEL Ở ĐÂY
    # =====================================================
    # "wavlm"  -> microsoft/wavlm-base
    # "hubert" -> facebook/hubert-base-ls960
    #
    # Gợi ý chạy:
    # scene + wavlm
    # for + wavlm
    # merged + wavlm
    #
    # scene + hubert
    # for + hubert
    # merged + hubert
    # =====================================================
    model_choice: str = "hubert"

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

    # Nếu True: đóng băng toàn bộ backbone, chỉ train classifier head
    freeze_backbone: bool = False

    # Nếu freeze_backbone=False:
    # 0 -> unfreeze toàn bộ backbone
    # 4 -> chỉ unfreeze 4 layer cuối
    unfreeze_last_n_layers: int = 4

    use_class_weight: bool = True

    early_stopping_patience: int = 3

    save_dir: str = "/kaggle/working"


CFG = Config()


# =========================================================
# 2) MODEL NAME + DATASET PATH
# =========================================================
def get_model_name(model_choice: str) -> str:
    model_choice = model_choice.lower()

    if model_choice == "wavlm":
        return "microsoft/wavlm-base"

    if model_choice == "hubert":
        return "facebook/hubert-base-ls960"

    raise ValueError(
        f"model_choice không hợp lệ: {model_choice}. "
        f"Chỉ nhận 'wavlm' hoặc 'hubert'."
    )


def get_dataset_root(cfg: Config) -> str:
    mode = cfg.dataset_mode.lower()

    if mode == "scene":
        return os.path.join(cfg.data_root, cfg.scene_name, cfg.scene_name)

    if mode == "for":
        return os.path.join(cfg.data_root, cfg.for_name, cfg.for_name)

    if mode == "merged":
        return os.path.join(cfg.data_root, cfg.merged_name, cfg.merged_name)

    raise ValueError(
        f"dataset_mode không hợp lệ: {cfg.dataset_mode}. "
        f"Chỉ nhận 'scene', 'for', hoặc 'merged'."
    )


CFG.model_name = get_model_name(CFG.model_choice)

DATASET_ROOT = get_dataset_root(CFG)
CFG.train_dir = os.path.join(DATASET_ROOT, "train_set")
CFG.val_dir = os.path.join(DATASET_ROOT, "eval_set")

os.makedirs(CFG.save_dir, exist_ok=True)

CFG.save_path = os.path.join(
    CFG.save_dir,
    f"best_{CFG.dataset_mode}_{CFG.model_choice}.pt",
)


# =========================================================
# 3) DEVICE + SEED
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Giúp kết quả ổn định hơn
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(CFG.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"Using device: {device}")


# =========================================================
# 4) DATA
# =========================================================
AUDIO_EXTENSIONS = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]


def find_audio_files(folder: str) -> List[str]:
    files = []

    for ext in AUDIO_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, ext)))

    return sorted(files)


def is_valid_audio_file(path: str) -> bool:
    try:
        wav, sr = torchaudio.load(path)

        if sr <= 0:
            return False

        if wav.numel() == 0:
            return False

        return True

    except Exception:
        return False


def collect_labeled_files(data_dir: str) -> List[Tuple[str, int]]:
    """
    Label:
    real = 0
    fake = 1
    """

    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")

    
    real_files_raw = find_audio_files(real_dir)
    fake_files_raw = find_audio_files(fake_dir)

    real_files = []
    fake_files = []

    skipped_real = 0
    skipped_fake = 0

    for path in real_files_raw:
        if is_valid_audio_file(path):
            real_files.append(path)
        else:
            skipped_real += 1
            print(f"[SKIP] Invalid real audio: {path}")

    for path in fake_files_raw:
        if is_valid_audio_file(path):
            fake_files.append(path)
        else:
            skipped_fake += 1
            print(f"[SKIP] Invalid fake audio: {path}")

    samples = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
    random.shuffle(samples)

    print(f"\nData directory: {data_dir}")
    print(f"Real files: {len(real_files)}")
    print(f"Fake files: {len(fake_files)}")
    print(f"Skipped invalid real files: {skipped_real}")
    print(f"Skipped invalid fake files: {skipped_fake}")
    print(f"Total files: {len(samples)}")

    return samples


def preprocess_audio(path: str, target_sr: int, clip_seconds: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(path)
    except Exception as e:
        raise RuntimeError(f"Lỗi đọc file audio: {path}\n{e}")

    # Stereo -> mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample về 16kHz
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = wav.squeeze(0)

    # Normalize tránh biên độ quá lớn
    wav = wav / (wav.abs().max() + 1e-9)

    target_len = target_sr * clip_seconds

    # Cắt hoặc pad về cùng độ dài
    if wav.numel() >= target_len:
        wav = wav[:target_len]
    else:
        wav = F.pad(wav, (0, target_len - wav.numel()))

    return wav


class DeepfakeAudioDataset(Dataset):
    def __init__(self, samples, target_sr, clip_seconds):
        self.samples = samples
        self.target_sr = target_sr
        self.clip_seconds = clip_seconds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        wav = preprocess_audio(
            path=path,
            target_sr=self.target_sr,
            clip_seconds=self.clip_seconds,
        )

        return {
            "waveform": wav,
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
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
        return_attention_mask=True,
    )

    return {
        "input_values": inputs["input_values"],
        "attention_mask": inputs.get("attention_mask"),
        "labels": labels,
    }


# =========================================================
# 5) MODEL
# =========================================================
class MeanStdPooling(nn.Module):
    def forward(self, hs, attention_mask=None):
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


class AudioDeepfakeDetector(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.model_name)

        hidden = self.backbone.config.hidden_size

        self.pool = MeanStdPooling()

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.num_classes),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hs = outputs.last_hidden_state

        feat_mask = None

        if attention_mask is not None and hasattr(
            self.backbone,
            "_get_feature_vector_attention_mask",
        ):
            feat_mask = self.backbone._get_feature_vector_attention_mask(
                hs.shape[1],
                attention_mask,
            )

        emb = self.pool(hs, feat_mask)
        logits = self.classifier(emb)

        return logits


def configure_trainable_layers(model: AudioDeepfakeDetector, cfg: Config):
    # Đóng băng backbone trước
    for p in model.backbone.parameters():
        p.requires_grad = False

    if cfg.freeze_backbone:
        print("Backbone frozen. Chỉ train classifier head.")
    else:
        if cfg.unfreeze_last_n_layers == 0:
            print("Unfreeze toàn bộ backbone.")
            for p in model.backbone.parameters():
                p.requires_grad = True

        elif hasattr(model.backbone, "encoder") and hasattr(
            model.backbone.encoder,
            "layers",
        ):
            layers = model.backbone.encoder.layers
            total_layers = len(layers)
            n_unfreeze = min(cfg.unfreeze_last_n_layers, total_layers)

            print(f"Unfreeze {n_unfreeze}/{total_layers} layer cuối của backbone.")

            for layer in layers[-n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True

        else:
            print("Không tìm thấy encoder.layers. Unfreeze toàn bộ backbone.")
            for p in model.backbone.parameters():
                p.requires_grad = True

    # Classifier luôn được train
    for p in model.classifier.parameters():
        p.requires_grad = True


def count_trainable_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")


# =========================================================
# 6) METRICS
# =========================================================
def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "specificity": specificity,
    }


# =========================================================
# 7) TRAIN LOOP
# =========================================================
def run_epoch(model, loader, optimizer, criterion, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for batch_idx, batch in enumerate(loader):
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device)

        mask = batch["attention_mask"]
        if mask is not None:
            mask = mask.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x, mask)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * y.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true_all.extend(y.detach().cpu().tolist())
        y_pred_all.extend(preds.detach().cpu().tolist())
        y_prob_all.extend(probs.detach().cpu().tolist())

    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def build_optimizer(model: AudioDeepfakeDetector, cfg: Config):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": cfg.lr_backbone,
            },
            {
                "params": head_params,
                "lr": cfg.lr_head,
            },
        ],
        weight_decay=cfg.weight_decay,
    )

    return optimizer


# =========================================================
# 8) MAIN
# =========================================================
def main():
    print("\n" + "=" * 70)
    print("CONFIG")
    print("=" * 70)
    print(f"Dataset mode: {CFG.dataset_mode}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Train dir   : {CFG.train_dir}")
    print(f"Eval dir    : {CFG.val_dir}")
    print(f"Model choice: {CFG.model_choice}")
    print(f"Model name  : {CFG.model_name}")
    print(f"Save path   : {CFG.save_path}")
    print(f"Clip seconds: {CFG.clip_seconds}")
    print(f"Batch size  : {CFG.batch_size}")
    print(f"Epochs      : {CFG.num_epochs}")

    train_samples = collect_labeled_files(CFG.train_dir)
    val_samples = collect_labeled_files(CFG.val_dir)

    if len(train_samples) == 0:
        raise RuntimeError(
            f"Không tìm thấy file train nào trong {CFG.train_dir}. "
            f"Hãy kiểm tra lại dataset path."
        )

    if len(val_samples) == 0:
        raise RuntimeError(
            f"Không tìm thấy file eval nào trong {CFG.val_dir}. "
            f"Hãy kiểm tra lại dataset path."
        )

    train_real = sum(1 for _, y in train_samples if y == 0)
    train_fake = sum(1 for _, y in train_samples if y == 1)
    val_real = sum(1 for _, y in val_samples if y == 0)
    val_fake = sum(1 for _, y in val_samples if y == 1)

    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Train real : {train_real}")
    print(f"Train fake : {train_fake}")
    print(f"Val real   : {val_real}")
    print(f"Val fake   : {val_fake}")
    print(f"Train total: {len(train_samples)}")
    print(f"Val total  : {len(val_samples)}")

    train_total = len(train_samples)
    val_total = len(val_samples)
    all_total = train_total + val_total

    print(f"Train ratio: {train_total / all_total:.4f}")
    print(f"Val ratio  : {val_total / all_total:.4f}")

    train_loader = DataLoader(
        DeepfakeAudioDataset(
            samples=train_samples,
            target_sr=CFG.target_sr,
            clip_seconds=CFG.clip_seconds,
        ),
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=train_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        DeepfakeAudioDataset(
            samples=val_samples,
            target_sr=CFG.target_sr,
            clip_seconds=CFG.clip_seconds,
        ),
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=train_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = AudioDeepfakeDetector(CFG).to(device)

    configure_trainable_layers(model, CFG)
    count_trainable_params(model)

    # =====================================================
    # Loss
    # =====================================================
    if CFG.use_class_weight:
        counts = torch.tensor(
            [
                train_real,
                train_fake,
            ],
            dtype=torch.float32,
        )

        weights = counts.sum() / (len(counts) * counts)

        print("\nClass weights:")
        print(f"real weight: {weights[0].item():.4f}")
        print(f"fake weight: {weights[1].item():.4f}")

        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, CFG)

    best_f1 = -1.0
    best_auc = -1.0

    patience = CFG.early_stopping_patience
    counter = 0

    print("\n" + "=" * 70)
    print("START TRAINING")
    print("=" * 70)

    for epoch in range(CFG.num_epochs):
        train_m = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            train=True,
        )

        val_m = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            train=False,
        )

        print("\n" + "-" * 70)
        print(f"Epoch {epoch + 1}/{CFG.num_epochs}")
        print("-" * 70)

        print(
            f"Train | "
            f"Loss: {train_m['loss']:.4f} | "
            f"Acc: {train_m['acc']:.4f} | "
            f"Precision: {train_m['precision']:.4f} | "
            f"Recall: {train_m['recall']:.4f} | "
            f"F1: {train_m['f1']:.4f} | "
            f"AUC: {train_m['auc']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_m['loss']:.4f} | "
            f"Acc: {val_m['acc']:.4f} | "
            f"Precision: {val_m['precision']:.4f} | "
            f"Recall: {val_m['recall']:.4f} | "
            f"F1: {val_m['f1']:.4f} | "
            f"AUC: {val_m['auc']:.4f}"
        )

        print(
            f"CM    | "
            f"TN: {val_m['tn']} | "
            f"FP: {val_m['fp']} | "
            f"FN: {val_m['fn']} | "
            f"TP: {val_m['tp']} | "
            f"Specificity: {val_m['specificity']:.4f}"
        )

        improved = False

        if val_m["f1"] > best_f1:
            improved = True
        elif val_m["f1"] == best_f1 and val_m["auc"] > best_auc:
            improved = True

        if improved:
            best_f1 = val_m["f1"]
            best_auc = val_m["auc"]
            counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": CFG.__dict__,
                    "dataset_mode": CFG.dataset_mode,
                    "model_choice": CFG.model_choice,
                    "model_name": CFG.model_name,
                    "best_f1": best_f1,
                    "best_auc": best_auc,
                    "epoch": epoch + 1,
                },
                CFG.save_path,
            )

            print(f"-> Save best model: {CFG.save_path}")
        else:
            counter += 1
            print(f"-> EarlyStopping: {counter}/{patience}")

            if counter >= patience:
                print("-> Early stopping!")
                break

    print("\nTraining done!")
    print(f"Best F1 : {best_f1:.4f}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Saved at: {CFG.save_path}")


if __name__ == "__main__":
    main()