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
# Dataclass này dùng để gom toàn bộ cấu hình của thí nghiệm vào một chỗ.
# Nhờ vậy, khi muốn đổi dataset, đổi backbone, đổi số epoch, đổi learning rate...
# ta chỉ cần sửa tại đây.
# =========================================================

@dataclass
class Config:
    # -----------------------------------------------------
    # Đường dẫn dữ liệu train.
    # Cấu trúc thư mục được kỳ vọng là:
    #
    # data/train_data_v1/
    # ├── real/
    # │   ├── a.wav
    # │   ├── b.wav
    # │   └── ...
    # └── fake/
    #     ├── c.wav
    #     ├── d.wav
    #     └── ...
    #
    # real  -> label 0
    # fake  -> label 1
    # -----------------------------------------------------
    train_dir: str = "data/train_data_v1"

    # -----------------------------------------------------
    # Backbone pretrained.
    # Đây là phần "mạng chính" hay "bộ não chính" dùng để trích đặc trưng âm thanh.
    #
    # Có thể đổi giữa:
    # - "microsoft/wavlm-base"
    # - "facebook/wav2vec2-base"
    #
    # Khi đổi model_name, tức là bạn đang đổi backbone để so sánh.
    # -----------------------------------------------------
    model_name: str = "microsoft/wavlm-base"
    # model_name: str = "facebook/wav2vec2-base"

    # -----------------------------------------------------
    # target_sr: sample rate chuẩn mà mô hình mong đợi.
    # Hầu hết speech model pretrained thường dùng 16kHz.
    #
    # clip_seconds: độ dài cố định của mỗi audio sau preprocess.
    # Ví dụ 4 giây => 16000 * 4 = 64000 samples.
    # -----------------------------------------------------
    target_sr: int = 16000
    clip_seconds: int = 4

    # -----------------------------------------------------
    # batch_size: số lượng sample được đưa vào model cùng lúc.
    # num_epochs: số vòng lặp huấn luyện qua toàn bộ train set.
    #
    # lr_backbone: learning rate cho backbone pretrained
    # lr_head: learning rate cho classifier head mới thêm vào
    #
    # weight_decay: regularization giúp giảm overfit
    #
    # val_ratio: tỉ lệ tách validation từ train set
    #
    # seed: để cố định random, giúp kết quả dễ lặp lại
    #
    # num_workers: số process phụ để load data
    #   - 0: an toàn, dễ debug
    #   - >0: load nhanh hơn trên máy phù hợp
    # -----------------------------------------------------
    batch_size: int = 4
    num_epochs: int = 10
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 0

    # -----------------------------------------------------
    # num_classes = 2 vì đây là bài toán phân loại nhị phân:
    # 0 = real
    # 1 = fake
    #
    # dropout: giảm overfitting ở classifier head
    # head_hidden_dim: số chiều hidden layer của MLP head
    # -----------------------------------------------------
    num_classes: int = 2
    dropout: float = 0.3
    head_hidden_dim: int = 256

    # -----------------------------------------------------
    # pooling: cách gộp hidden states theo thời gian
    #
    # Backbone trả ra đặc trưng dạng chuỗi [B, T, H]
    # nhưng classifier head cần 1 vector cho mỗi audio.
    #
    # Vì vậy cần pooling để biến [B, T, H] -> [B, D]
    #
    # Các lựa chọn:
    # - "mean"      : lấy trung bình theo thời gian
    # - "mean_std"  : nối mean và std
    # - "attention" : attention pooling có trọng số học được
    # -----------------------------------------------------
    pooling: str = "mean_std"   # "mean" | "mean_std" | "attention"

    # -----------------------------------------------------
    # Chiến lược fine-tuning:
    #
    # freeze_backbone = True:
    #   chỉ train classifier head, backbone không update
    #
    # freeze_backbone = False:
    #   cho phép train một phần hoặc toàn bộ backbone
    #
    # unfreeze_last_n_layers:
    #   - 4  -> chỉ mở 4 layer cuối của backbone
    #   - 0  -> mở toàn bộ backbone (full fine-tune)
    # -----------------------------------------------------
    freeze_backbone: bool = False
    unfreeze_last_n_layers: int = 4

    # -----------------------------------------------------
    # use_class_weight:
    #   nếu real/fake lệch nhau nhiều, loss sẽ được gán weight
    #   để model không thiên vị lớp nhiều dữ liệu hơn.
    #
    # save_path:
    #   nơi lưu model tốt nhất sau khi train
    # -----------------------------------------------------
    use_class_weight: bool = True
    save_path: str = "best_model.pt"

# Tạo object config để dùng xuyên suốt file
CFG = Config()


# =========================================================
# 2) DEVICE + SEED
# =========================================================
# Phần này:
# - cố định random
# - xác định chạy trên GPU hay CPU
# =========================================================

def set_seed(seed: int):
    """
    Cố định tính ngẫu nhiên để kết quả train ổn định hơn,
    dễ tái hiện hơn giữa các lần chạy.
    """
    random.seed(seed)                   # random của Python
    torch.manual_seed(seed)             # random của PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)    # random của PyTorch (GPU)


set_seed(CFG.seed)

# Nếu máy có CUDA GPU thì dùng GPU, không thì fallback về CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 3) DATA
# =========================================================
# Phần này lo:
# - đọc file trong thư mục
# - preprocess audio
# - xây Dataset cho PyTorch
# =========================================================

def collect_labeled_files(train_dir: str) -> List[Tuple[str, int]]:
    """
    Quét toàn bộ file .wav trong:
    - train_dir/real
    - train_dir/fake

    Sau đó gán nhãn:
    - real -> 0
    - fake -> 1

    Return:
        List[(path, label)]
        Ví dụ:
        [
            ("data/train_data_v1/real/a.wav", 0),
            ("data/train_data_v1/fake/b.wav", 1),
            ...
        ]
    """

    # Lấy tất cả file WAV trong thư mục real
    real_files = glob.glob(os.path.join(train_dir, "real", "*.wav"))
    # Lấy tất cả file WAV trong thư mục fake
    fake_files = glob.glob(os.path.join(train_dir, "fake", "*.wav"))

    # Tạo list sample với label tương ứng
    samples = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]

    # Trộn ngẫu nhiên thứ tự sample
    random.shuffle(samples)

    return samples


def preprocess_audio(path: str, target_sr: int, clip_seconds: int) -> torch.Tensor:
    """
    Đọc và chuẩn hóa audio để đưa vào model.

    Các bước:
    1) Load audio bằng torchaudio
    2) Nếu audio nhiều channel -> gộp thành mono
    3) Nếu sample rate khác 16k -> resample về 16k
    4) Bỏ chiều channel [1, T] -> [T]
    5) Normalize biên độ về khoảng gần [-1, 1]
    6) Cắt hoặc pad để mọi audio có cùng độ dài

    Return:
        waveform 1D tensor shape [T]
    """

    # torchaudio.load trả về:
    # - wav: tensor shape [C, T]
    # - sr : sample rate gốc
    wav, sr = torchaudio.load(path)  # [C, T]

    # Nếu audio có nhiều channel (ví dụ stereo 2 kênh),
    # ta lấy trung bình các channel để thành mono.
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Nếu sample rate gốc khác target_sr, cần resample.
    # Ví dụ file 28kHz sẽ được chuyển về 16kHz.
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # Sau khi đã mono, wav thường có shape [1, T].
    # squeeze(0) để bỏ chiều channel -> [T]
    wav = wav.squeeze(0)

    # Normalize biên độ:
    # chia toàn bộ waveform cho giá trị tuyệt đối lớn nhất.
    # +1e-9 để tránh chia cho 0 nếu audio toàn số 0.
    wav = wav / (wav.abs().max() + 1e-9)

    # Số sample mục tiêu = sample rate * số giây
    # 16kHz * 4 giây = 64000 sample
    target_len = target_sr * clip_seconds

    # Nếu audio dài hơn target_len -> cắt bớt
    if wav.numel() >= target_len:
        wav = wav[:target_len]
    else:
        # Nếu audio ngắn hơn -> pad thêm số 0 ở cuối
        wav = F.pad(wav, (0, target_len - wav.numel()))

    return wav


class DeepfakeTrainDataset(Dataset):
    """
    Custom Dataset cho bài toán audio real/fake.

    Dataset này lưu danh sách sample dạng:
    [
        (path_1, label_1),
        (path_2, label_2),
        ...
    ]

    Khi DataLoader yêu cầu sample thứ i,
    __getitem__ sẽ:
    - đọc file audio
    - preprocess
    - trả về dictionary gồm waveform + label
    """

    def __init__(self, samples: List[Tuple[str, int]], target_sr: int, clip_seconds: int):
        self.samples = samples
        self.target_sr = target_sr
        self.clip_seconds = clip_seconds

    def __len__(self):
        """
        Trả về tổng số sample trong dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Trả về 1 sample tại vị trí idx.

        Output là dictionary:
        {
            "path": path file,
            "waveform": tensor audio đã preprocess,
            "label": tensor nhãn
        }
        """
        path, label = self.samples[idx]

        # Đọc + preprocess audio
        wav = preprocess_audio(path, self.target_sr, self.clip_seconds)
        return {
            "path": path,
            "waveform": wav,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------
# Feature extractor tương ứng với backbone pretrained.
#
# Vai trò:
# - chuẩn hóa input waveform
# - tạo input_values / attention_mask đúng format model mong đợi
#
# Lưu ý:
# - feature_extractor KHÔNG phải backbone
# - backbone là AutoModel
# ---------------------------------------------------------
feature_extractor = AutoFeatureExtractor.from_pretrained(CFG.model_name)


def train_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate_fn cho DataLoader.

    Tại sao cần collate_fn riêng?
    Vì batch là list nhiều sample:
        [
            {"waveform": ..., "label": ...},
            {"waveform": ..., "label": ...},
            ...
        ]

    Ta cần gộp chúng thành 1 batch tensor để model xử lý.

    Các bước:
    1) Lấy waveform từ từng sample
    2) Gọi feature_extractor để tạo input_values / attention_mask
    3) Stack labels thành tensor
    """

    # feature_extractor thường làm việc tốt với list numpy,
    # nên ta chuyển waveform tensor -> numpy
    waveforms = [item["waveform"].numpy() for item in batch]

    # Stack toàn bộ label thành tensor shape [B]
    labels = torch.stack([item["label"] for item in batch])

    # Feature extractor sẽ:
    # - nhận list waveform
    # - chuẩn hóa thành input_values
    # - tạo attention_mask nếu cần
    inputs = feature_extractor(
        waveforms,
        sampling_rate=CFG.target_sr,
        return_tensors="pt",
        padding=True,
    )

    return {
        "input_values": inputs["input_values"],             # tensor input cho backbone
        "attention_mask": inputs.get("attention_mask"),     # mask đánh dấu phần hợp lệ
        "labels": labels,                                   # tensor nhãn của batch  
        "paths": [item["path"] for item in batch],          # nhãn ground truth
    }


# =========================================================
# 4) POOLING
# =========================================================
# Backbone trả ra đặc trưng theo chuỗi thời gian:
#   [B, T_feat, H]
#
# Nhưng classifier head cần 1 vector cho mỗi clip.
# Vì vậy ta phải "gộp theo thời gian" bằng pooling.
# =========================================================

class MeanPooling(nn.Module):
    """
    Pooling bằng trung bình theo thời gian.

    Nếu có attention_mask:
    - chỉ lấy trung bình trên các frame hợp lệ
    - bỏ qua phần padding
    """
    def forward(self, hs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # hs shape: [B, T, H]
        if attention_mask is None:
            # Không có mask -> mean toàn bộ theo chiều thời gian
            return hs.mean(dim=1)

        # attention_mask shape gốc: [B, T]
        # thêm chiều cuối -> [B, T, 1] để nhân broadcast với hs
        mask = attention_mask.unsqueeze(-1).float()

        # Cộng các frame hợp lệ
        summed = (hs * mask).sum(dim=1)

        # Số frame hợp lệ ở mỗi sample
        denom = mask.sum(dim=1).clamp(min=1e-6)

        # Mean có mask
        return summed / denom


class MeanStdPooling(nn.Module):
    """
    Pooling bằng:
    - mean theo thời gian
    - std theo thời gian
    rồi nối (concatenate) lại.

    Output sẽ có số chiều gấp đôi hidden size:
    [B, 2H]

    Kiểu pooling này hay dùng trong speaker/audio tasks
    vì nó giữ được cả:
    - xu hướng trung bình
    - mức độ biến thiên của đặc trưng
    """
    def forward(self, hs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            mean = hs.mean(dim=1)
            std = hs.std(dim=1)
            return torch.cat([mean, std], dim=1)

        mask = attention_mask.unsqueeze(-1).float()
        lengths = mask.sum(dim=1).clamp(min=1e-6)

        # Mean có mask
        mean = (hs * mask).sum(dim=1) / lengths

        # Mean có mask
        var = (((hs - mean.unsqueeze(1)) * mask) ** 2).sum(dim=1) / lengths

        # Std = sqrt(var)
        std = torch.sqrt(var + 1e-6)

        # Nối mean và std theo chiều feature
        return torch.cat([mean, std], dim=1)


class AttentionPooling(nn.Module):
    """
    Pooling có trọng số học được.

    Ý tưởng:
    - không phải frame nào cũng quan trọng như nhau
    - mô hình sẽ học tự gán trọng số lớn hơn cho frame quan trọng

    Cách làm:
    1) Mỗi frame [H] đi qua Linear(H -> 1) để tạo score
    2) Softmax score theo thời gian -> attention weights
    3) Tính weighted sum của hidden states
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # hs shape: [B, T, H]

        # Mỗi frame đi qua Linear -> score shape [B, T, 1]
        # squeeze(-1) -> [B, T]
        scores = self.attn(hs).squeeze(-1)

        # Nếu có mask, các frame padding sẽ bị gán -inf
        # để sau softmax trọng số của chúng gần như bằng 0
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax theo thời gian để ra attention weights
        # shape: [B, T, 1]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)

        # Weighted sum theo thời gian -> [B, H]
        pooled = (hs * weights).sum(dim=1)
        return pooled


# =========================================================
# 5) MODEL
# =========================================================
# Đây là mô hình hoàn chỉnh cho bài toán real/fake:
#
# input_values -> backbone -> hidden states -> pooling -> classifier -> logits
# =========================================================

class AudioDeepfakeDetector(nn.Module):
    """
    Model hoàn chỉnh gồm 3 phần:

    1) Backbone pretrained:
       - WavLM hoặc wav2vec2
       - nhiệm vụ: trích đặc trưng âm thanh

    2) Pooling:
       - gộp chuỗi hidden states [B, T, H] thành 1 vector [B, D]

    3) Classifier head:
       - MLP
       - nhiệm vụ: phân loại real / fake
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        dropout: float = 0.3,
        head_hidden_dim: int = 256,
        pooling: str = "mean_std",
    ):
        super().__init__()

        # -------------------------------------------------
        # Load backbone pretrained từ Hugging Face
        # Ví dụ:
        #   microsoft/wavlm-base
        #   facebook/wav2vec2-base
        # -------------------------------------------------
        self.backbone = AutoModel.from_pretrained(model_name)

        # hidden_size là số chiều đặc trưng mà backbone sinh ra
        # Ví dụ WavLM-base thường có H = 768
        hidden_size = self.backbone.config.hidden_size

        # -------------------------------------------------
        # Chọn pooling strategy
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Classifier head:
        # pooled vector -> hidden layer -> output 2 lớp
        #
        # Output cuối là logits, chưa qua softmax
        # -------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Forward pass của model.

        Input:
            input_values  : tensor waveform đã qua feature_extractor
            attention_mask: mask của input_values

        Output:
            logits: điểm số thô cho 2 lớp
            emb   : embedding sau pooling
        """

        # ---------------------------------------------
        # 1) Forward qua backbone pretrained
        # ---------------------------------------------
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        # last_hidden_state shape: [B, T_feat, H]
        # Trong đó:
        # - B: batch size
        # - T_feat: số time steps sau backbone
        # - H: hidden size
        hs = outputs.last_hidden_state # [B, T_feat, H]

        # ---------------------------------------------
        # 2) Chuyển attention_mask từ input level
        #    sang feature level
        #
        # Vì backbone thường có conv frontend làm giảm chiều thời gian,
        # nên attention_mask gốc [B, T_input] không còn khớp trực tiếp
        # với hs [B, T_feat, H].
        #
        # _get_feature_vector_attention_mask giúp map mask về đúng T_feat.
        # ---------------------------------------------
        feature_attention_mask = None
        if attention_mask is not None:
            feature_attention_mask = self.backbone._get_feature_vector_attention_mask(
                hs.shape[1], attention_mask
            )

        # ---------------------------------------------
        # 3) Pooling để gộp chuỗi hidden states thành
        #    1 vector embedding cho mỗi audio
        # ---------------------------------------------
        emb = self.pool(hs, feature_attention_mask)

        # ---------------------------------------------
        # 4) Classifier head -> logits
        # ---------------------------------------------
        logits = self.classifier(emb)

        return logits, emb


def configure_trainable_layers(model: AudioDeepfakeDetector, freeze_backbone: bool, unfreeze_last_n_layers: int):
    """
    Thiết lập layer nào được train, layer nào bị freeze.

    Các chế độ:
    1) freeze_backbone=True:
       - toàn bộ backbone bị freeze
       - chỉ train head / pooling có parameter

    2) freeze_backbone=False và unfreeze_last_n_layers=4:
       - chỉ mở 4 layer cuối của backbone

    3) freeze_backbone=False và unfreeze_last_n_layers=0:
       - mở toàn bộ backbone (full fine-tune)
    """

    # Mặc định freeze toàn bộ backbone trước
    for p in model.backbone.parameters():
        p.requires_grad = False

    if not freeze_backbone:
        # Nếu unfreeze_last_n_layers = 0 -> full fine-tune
        if unfreeze_last_n_layers == 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
        else:
            # Cố gắng chỉ mở các layer cuối của encoder nếu backbone hỗ trợ
            if hasattr(model.backbone, "encoder") and hasattr(model.backbone.encoder, "layers"):
                layers = model.backbone.encoder.layers

                # Mở N layer cuối
                for layer in layers[-unfreeze_last_n_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

                # Có thể mở thêm feature_projection nếu tồn tại
                if hasattr(model.backbone, "feature_projection"):
                    for p in model.backbone.feature_projection.parameters():
                        p.requires_grad = True
            else:
                # Nếu kiến trúc không có encoder.layers như kỳ vọng,
                # fallback: mở toàn bộ backbone
                for p in model.backbone.parameters():
                    p.requires_grad = True

    # Classifier head luôn phải train
    for p in model.classifier.parameters():
        p.requires_grad = True

    # Pooling nếu có parameter (ví dụ attention pooling) cũng phải train
    for p in model.pool.parameters():
        p.requires_grad = True


# =========================================================
# 6) METRICS
# =========================================================
# Tính các chỉ số đánh giá cho bài toán phân loại nhị phân.
# =========================================================

def compute_metrics(y_true, y_pred, y_prob):
    """
    Tính các metric:
    - Accuracy
    - Precision
    - Recall
    - F1
    - AUC

    Input:
        y_true: list nhãn thật
        y_pred: list nhãn dự đoán (0/1)
        y_prob: list xác suất lớp fake
    """

    # Accuracy = tỉ lệ dự đoán đúng
    acc = accuracy_score(y_true, y_pred)

    # Precision / Recall / F1 cho bài toán nhị phân
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # AUC sử dụng xác suất dự đoán của lớp positive (ở đây là fake)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        # Nếu có lỗi (ví dụ val set toàn 1 lớp) thì trả NaN
        auc = float("nan")

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def train_one_epoch(model, loader, optimizer, criterion):
    """
    Train model trong 1 epoch.

    Các bước của mỗi batch:
    1) lấy input và label
    2) forward qua model
    3) tính loss
    4) backward
    5) optimizer step
    6) lưu y_true / y_pred / y_prob để tính metric cuối epoch
    """
    model.train()   # bật chế độ train (dropout hoạt động)
    total_loss = 0.0

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for batch in loader:
        # ---------------------------------------------
        # Chuyển input / label lên device (GPU hoặc CPU)
        # ---------------------------------------------
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Xóa gradient cũ trước khi backward batch mới
        optimizer.zero_grad()

        # Forward
        logits, _ = model(input_values=input_values, attention_mask=attention_mask)

        # Tính loss giữa logits và labels
        loss = criterion(logits, labels)

        # Backpropagation: tính gradient
        loss.backward()

        # Cập nhật trọng số
        optimizer.step()

        # Cộng dồn loss để cuối epoch lấy trung bình
        total_loss += loss.item() * labels.size(0)

        # Softmax logits để ra xác suất 2 lớp
        # [:, 1] = xác suất lớp fake
        probs = torch.softmax(logits, dim=1)[:, 1]

        # Class dự đoán = class có score lớn nhất
        preds = torch.argmax(logits, dim=1)

        # Lưu để cuối epoch tính metric
        y_true_all.extend(labels.detach().cpu().tolist())
        y_pred_all.extend(preds.detach().cpu().tolist())
        y_prob_all.extend(probs.detach().cpu().tolist())

    # Loss trung bình trên toàn dataset
    avg_loss = total_loss / len(loader.dataset)

    # Tính metric
    metrics = compute_metrics(y_true_all, y_pred_all, y_prob_all)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    """
    Validate model trong 1 epoch.

    Khác với train:
    - không backward
    - không optimizer.step()
    - chỉ forward và tính metric

    @torch.no_grad() giúp:
    - tiết kiệm bộ nhớ
    - chạy nhanh hơn
    """
    model.eval()    # bật chế độ eval (dropout tắt)
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

        # Chỉ forward, không train
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
# Đây là hàm điều phối toàn bộ quá trình huấn luyện.
# =========================================================

def main():
    """
    Luồng chính:
    1) Đọc dữ liệu
    2) Tạo dataset
    3) Split train/val
    4) Tạo dataloader
    5) Tạo model
    6) Cấu hình freeze/unfreeze
    7) Tạo loss và optimizer
    8) Train nhiều epoch
    9) Lưu best model theo F1 validation
    """
    print(f"Using device: {device}")
    print(f"Backbone: {CFG.model_name}")
    print(f"Pooling: {CFG.pooling}")

    # -----------------------------------------------------
    # B1. Đọc toàn bộ sample từ thư mục train
    # -----------------------------------------------------
    samples = collect_labeled_files(CFG.train_dir)

    # Nếu không tìm thấy file nào thì báo lỗi rõ ràng
    if len(samples) == 0:
        raise ValueError(f"Không tìm thấy file wav trong {CFG.train_dir}/real hoặc {CFG.train_dir}/fake")

    # Đếm số lượng real và fake để log / tính class weights
    num_real = sum(1 for _, y in samples if y == 0)
    num_fake = sum(1 for _, y in samples if y == 1)

    print(f"Total train samples: {len(samples)}")
    print(f"Real: {num_real}")
    print(f"Fake: {num_fake}")

    # -----------------------------------------------------
    # B2. Tạo full dataset
    # -----------------------------------------------------
    full_dataset = DeepfakeTrainDataset(samples, CFG.target_sr, CFG.clip_seconds)

    # -----------------------------------------------------
    # B3. Chia train / validation
    # -----------------------------------------------------
    val_size = int(len(full_dataset) * CFG.val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CFG.seed),
    )

    # -----------------------------------------------------
    # B4. Tạo DataLoader
    # -----------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,                   # train thì shuffle
        num_workers=CFG.num_workers,
        collate_fn=train_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,                 # val thì không cần shuffle
        num_workers=CFG.num_workers,
        collate_fn=train_collate_fn,
    )

    # -----------------------------------------------------
    # B5. Tạo model và đưa lên device
    # -----------------------------------------------------
    model = AudioDeepfakeDetector(
        model_name=CFG.model_name,
        num_classes=CFG.num_classes,
        dropout=CFG.dropout,
        head_hidden_dim=CFG.head_hidden_dim,
        pooling=CFG.pooling,
    ).to(device)

    # -----------------------------------------------------
    # B6. Thiết lập fine-tuning strategy
    # -----------------------------------------------------
    configure_trainable_layers(
        model,
        freeze_backbone=CFG.freeze_backbone,
        unfreeze_last_n_layers=CFG.unfreeze_last_n_layers,
    )

    # -----------------------------------------------------
    # B7. Tạo loss function
    # -----------------------------------------------------
    if CFG.use_class_weight:
        # Tính class weights theo tần suất nghịch đảo:
        # lớp ít dữ liệu -> weight cao hơn
        class_counts = torch.tensor([num_real, num_fake], dtype=torch.float32)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # B8. Tách tham số backbone và head để gán learning rate khác nhau
    # -----------------------------------------------------
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        # Nếu parameter không train thì bỏ qua
        if not p.requires_grad:
            continue

        # Parameter của backbone
        if name.startswith("backbone"):
            backbone_params.append(p)
        else:
            # classifier / pooling
            head_params.append(p)

    # AdamW optimizer với 2 nhóm parameter:
    # - backbone lr nhỏ
    # - head lr lớn hơn
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": CFG.lr_backbone},
            {"params": head_params, "lr": CFG.lr_head},
        ],
        weight_decay=CFG.weight_decay,
    )

    # -----------------------------------------------------
    # B9. Train nhiều epoch và lưu best model theo F1 val
    # -----------------------------------------------------
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

        # Nếu F1 validation tốt hơn trước đó -> lưu checkpoint
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), CFG.save_path)
            print(f"Saved best model to {CFG.save_path}")

    print(f"\nTraining xong. Best model đã lưu ở: {CFG.save_path}")


# 8) ENTRY POINT
# =========================================================
# Chỉ chạy main() khi file này được chạy trực tiếp:
#   python train.py
#
# Nếu file được import từ nơi khác thì main() sẽ không tự chạy.
# =========================================================
if __name__ == "__main__":
    main()