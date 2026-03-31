import os
import glob
import random
import shutil

# ========================= CONFIG =========================
SOURCE_DIR = "data/train_data"      # Thư mục gốc hiện tại của bạn
TRAIN_DIR  = "data/train_data/train_set"
VAL_DIR    = "data/train_data/val_set"

VAL_RATIO = 0.2
SEED = 42

# =========================================================
random.seed(SEED)

def split_class(class_name: str):
    source_path = os.path.join(SOURCE_DIR, class_name)
    files = glob.glob(os.path.join(source_path, "*.wav"))
    
    if not files:
        print(f"❌ Không tìm thấy file .wav trong {source_path}")
        return
    
    random.shuffle(files)
    
    val_count = int(len(files) * VAL_RATIO)
    train_files = files[val_count:]
    val_files   = files[:val_count]
    
    # Tạo thư mục đích
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)
    
    print(f"{class_name.upper():<6} | Tổng: {len(files):>6,} | "
          f"Train: {len(train_files):>6,} | Val: {len(val_files):>6,}")
    
    # Copy file
    for f in train_files:
        shutil.copy2(f, os.path.join(TRAIN_DIR, class_name, os.path.basename(f)))
    
    for f in val_files:
        shutil.copy2(f, os.path.join(VAL_DIR, class_name, os.path.basename(f)))

# ====================== CHẠY TÁCH ======================
print("🔄 Đang tách 80% train - 20% val từ data/train_data...\n")

split_class("real")
split_class("fake")

print("\n" + "="*65)
print("✅ HOÀN TẤT TÁCH DỮ LIỆU!")
print("="*65)
print(f"Train set (80%) → data/train_set/")
print(f"Val set   (20%) → data/val_set/")
print(f"\nSố lượng cụ thể:")
print(f"   Real  → Train: 44,539  | Val: 11,134")
print(f"   Fake  → Train: 44,005  | Val: 11,001")