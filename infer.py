import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoFeatureExtractor


# =========================================================
# 1) CONFIG
# =========================================================

@dataclass
class Config:
    model_name: str = "microsoft/wavlm-base"
    target_sr: int = 16000
    clip_seconds: int = 4

    num_classes: int = 2
    dropout: float = 0.3
    head_hidden_dim: int = 256
    pooling: str = "mean_std"

    save_path: str = "best_model.pt"


CFG = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 2) PREPROCESS
# =========================================================

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


# =========================================================
# 3) MODEL
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
        hs = outputs.last_hidden_state
        feature_attention_mask = None
        if attention_mask is not None:
            feature_attention_mask = self.backbone._get_feature_vector_attention_mask(
                hs.shape[1], attention_mask
            )

        emb = self.pool(hs, feature_attention_mask)
        logits = self.classifier(emb)
        return logits, emb


# =========================================================
# 4) LOAD MODEL 1 LẦN
# =========================================================

feature_extractor = AutoFeatureExtractor.from_pretrained(CFG.model_name)

_model = AudioDeepfakeDetector(
    model_name=CFG.model_name,
    num_classes=CFG.num_classes,
    dropout=CFG.dropout,
    head_hidden_dim=CFG.head_hidden_dim,
    pooling=CFG.pooling,
).to(device)

if not os.path.exists(CFG.save_path):
    raise FileNotFoundError(f"Không tìm thấy model đã train: {CFG.save_path}")

_model.load_state_dict(torch.load(CFG.save_path, map_location=device))
_model.eval()


# =========================================================
# 5) INFER 1 FILE
# =========================================================

@torch.no_grad()
def infer(path: str):
    wav = preprocess_audio(path, CFG.target_sr, CFG.clip_seconds)

    inputs = feature_extractor(
        [wav.numpy()],
        sampling_rate=CFG.target_sr,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    logits, _ = _model(input_values=input_values, attention_mask=attention_mask)
    probs = torch.softmax(logits, dim=1)[0]

    pred_label = torch.argmax(probs).item()
    fake_prob = probs[1].item()
    real_prob = probs[0].item()

    label_name = "fake" if pred_label == 1 else "real"

    return {
        "label": label_name,
        "pred_label": pred_label,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
    }