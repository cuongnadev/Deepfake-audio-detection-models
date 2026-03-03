import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# ==============================
# 1) CẤU HÌNH CHUNG
# ==============================

# WavLM-base: encoder pretrained để trích đặc trưng âm thanh (chưa phải detector real/fake)
MODEL_NAME = "microsoft/wavlm-base"

# Chuẩn sample-rate cho đa số model speech pretrained
TARGET_SR = 16000

# Cố định độ dài clip để đưa vào model (giây)
CLIP_SECONDS = 4

# Chọn thiết bị chạy
device = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# 2) LOAD FEATURE EXTRACTOR + MODEL
# ==============================

# FeatureExtractor: xử lý waveform -> input_values (KHÔNG dùng tokenizer)
# Đây là thứ ta cần cho audio classification/inference kiểu speech encoder.
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

# AutoModel: load encoder WavLM
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Chế độ inference (tắt dropout, v.v.)
model.eval()


# ==============================
# 3) PREPROCESS AUDIO
# ==============================
def preprocess(path: str) -> torch.Tensor:
    """
    Đọc và chuẩn hóa audio:
    - Load audio (wav, sr)
    - Stereo -> mono
    - Resample về 16kHz
    - Normalize amplitude (tránh clip)
    - Cắt/pad về đúng CLIP_SECONDS
    Trả về: waveform 1D tensor shape [time]
    """

    # Load audio: wav shape [channels, time], sr là sample rate gốc
    wav, sr = torchaudio.load(path)

    # Stereo/multi-channel -> mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample về TARGET_SR nếu cần
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    # Bỏ channel dimension: [1, time] -> [time]
    wav = wav.squeeze(0)

    # Normalize biên độ về [-1, 1] (tránh trường hợp file quá to/nhỏ)
    wav = wav / (wav.abs().max() + 1e-9)

    # Cắt/pad về đúng số samples
    target_len = TARGET_SR * CLIP_SECONDS
    if wav.numel() >= target_len:
        wav = wav[:target_len]
    else:
        wav = torch.nn.functional.pad(wav, (0, target_len - wav.numel()))

    return wav


# ==============================
# 4) INFERENCE: TRÍCH EMBEDDING
# ==============================
@torch.no_grad()
def infer(path: str) -> float:
    """
    Chạy 1 file audio qua WavLM:
    - preprocess waveform
    - feature_extractor -> input_values
    - model forward -> hidden states [B, T, H]
    - pooling (mean+std) -> embedding [B, 2H]
    - trả về score tạm thời (để test pipeline chạy OK)

    LƯU Ý: score này CHƯA phải real/fake.
    Muốn real/fake cần thêm classifier head + train với dữ liệu fake.
    """

    # 1) Chuẩn hóa audio
    wav = preprocess(path)

    # 2) Convert waveform thành input_values cho model
    # FeatureExtractor thường nhận numpy/list
    inputs = feature_extractor(
        wav.cpu().numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )

    # 3) Đưa input lên GPU nếu có
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4) Forward qua encoder
    out = model(**inputs)
    hs = out.last_hidden_state  # [B, T, H]

    # 5) Pooling: gộp theo thời gian để ra 1 vector/clip
    mean = hs.mean(dim=1)        # [B, H]
    std = hs.std(dim=1)          # [B, H]
    emb = torch.cat([mean, std], dim=1)  # [B, 2H]

    # 6) Score tạm để test (chỉ để thấy có output)
    score = emb.mean().item()
    return score