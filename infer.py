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
    # =====================================================
    # MODEL SELECTION
    # =====================================================
    # "hubert" -> facebook/hubert-base-ls960
    # "wavlm"  -> microsoft/wavlm-base
    # =====================================================
    model_choice: str = "hubert"

    # =====================================================
    # CHECKPOINT PATH
    # =====================================================
    # Use one of:
    # - best_merged_hubert.pt
    # - best_merged_wavlm.pt
    # =====================================================
    save_path: str = "best_merged_hubert.pt"

    target_sr: int = 16000
    clip_seconds: int = 4

    num_classes: int = 2
    dropout: float = 0.3
    head_hidden_dim: int = 256


CFG = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 2) MODEL NAME
# =========================================================
def get_model_name(model_choice: str) -> str:
    model_choice = model_choice.lower()

    if model_choice == "wavlm":
        return "microsoft/wavlm-base"

    if model_choice == "hubert":
        return "facebook/hubert-base-ls960"

    raise ValueError(
        f"Invalid model_choice: {model_choice}. "
        f"Expected 'wavlm' or 'hubert'."
    )


CFG.model_name = get_model_name(CFG.model_choice)


# =========================================================
# 3) PREPROCESS
# =========================================================
def preprocess_audio(path: str, target_sr: int, clip_seconds: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {path}\n{e}")

    # Stereo -> mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample to target sampling rate
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = wav.squeeze(0)

    # Normalize amplitude
    wav = wav / (wav.abs().max() + 1e-9)

    target_len = target_sr * clip_seconds

    # Crop or pad to fixed length
    if wav.numel() >= target_len:
        wav = wav[:target_len]
    else:
        wav = F.pad(wav, (0, target_len - wav.numel()))

    return wav


# =========================================================
# 4) MODEL
# =========================================================
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


class AudioDeepfakeDetector(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        hidden_size = self.backbone.config.hidden_size

        self.pool = MeanStdPooling()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.num_classes),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hs = outputs.last_hidden_state

        feature_attention_mask = None

        if attention_mask is not None and hasattr(
            self.backbone,
            "_get_feature_vector_attention_mask",
        ):
            feature_attention_mask = self.backbone._get_feature_vector_attention_mask(
                hs.shape[1],
                attention_mask,
            )

        emb = self.pool(hs, feature_attention_mask)
        logits = self.classifier(emb)

        return logits


# =========================================================
# 5) LOAD MODEL ONCE
# =========================================================
feature_extractor = AutoFeatureExtractor.from_pretrained(CFG.model_name)

_model = AudioDeepfakeDetector(CFG).to(device)

if not os.path.exists(CFG.save_path):
    raise FileNotFoundError(f"Trained checkpoint was not found: {CFG.save_path}")

checkpoint = torch.load(
    CFG.save_path,
    map_location=device,
    weights_only=False,
)

# New training checkpoint format:
# {
#   "model_state_dict": ...,
#   "config": ...,
#   ...
# }
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    # Fallback for old checkpoint format: direct state_dict
    state_dict = checkpoint

_model.load_state_dict(state_dict)
_model.eval()

print(f"Loaded model: {CFG.model_name}")
print(f"Loaded checkpoint: {CFG.save_path}")
print(f"Device: {device}")


# =========================================================
# 6) INFERENCE FOR ONE FILE
# =========================================================
@torch.no_grad()
def infer(path: str):
    wav = preprocess_audio(
        path=path,
        target_sr=CFG.target_sr,
        clip_seconds=CFG.clip_seconds,
    )

    inputs = feature_extractor(
        [wav.numpy()],
        sampling_rate=CFG.target_sr,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )

    input_values = inputs["input_values"].to(device)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    logits = _model(
        input_values=input_values,
        attention_mask=attention_mask,
    )

    probs = torch.softmax(logits, dim=1)[0]

    real_prob = probs[0].item()
    fake_prob = probs[1].item()

    pred_label = torch.argmax(probs).item()
    label_name = "real" if pred_label == 0 else "fake"

    return {
        "label": label_name,
        "pred_label": pred_label,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "confidence": max(real_prob, fake_prob),
        "model_choice": CFG.model_choice,
        "model_name": CFG.model_name,
        "checkpoint": CFG.save_path,
    }