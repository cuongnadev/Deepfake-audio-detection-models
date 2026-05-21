# CHECK DATASET STRUCTURE ON KAGGLE
# =========================================================
# Purpose:
# - Check whether the dataset directory has been mounted correctly on Kaggle.
# - Check whether the FoR, SceneFake, and Merged datasets exist.
# - Check whether each dataset contains train_set/eval_set.
# - Check whether each split contains fake/real folders.
# - Count the number of .wav and .mp3 audio files in each folder.
#
# Expected structure:
# deepfake-audio-dataset/
# ├── FoR/
# │   ├── train_set/
# │   │   ├── fake/
# │   │   └── real/
# │   └── eval_set/
# │       ├── fake/
# │       └── real/
# ├── SceneFake/
# │   ├── train_set/
# │   │   ├── fake/
# │   │   └── real/
# │   └── eval_set/
# │       ├── fake/
# │       └── real/
# └── Merged/
#     ├── train_set/
#     │   ├── fake/
#     │   └── real/
#     └── eval_set/
#         ├── fake/
#         └── real/
# =========================================================

import os

dataset_path = "/kaggle/input/datasets/anhcngnguyn/deepfake-audio-dataset"

print("Dataset path:", dataset_path)

if not os.path.exists(dataset_path):
    print("Dataset path was not found.")
else:
    print("\nDataset directory contents:")
    print(os.listdir(dataset_path))

    dataset_names = ["FoR/FoR", "SceneFake/SceneFake", "Merged/Merged"]

    for dataset_name in dataset_names:
        dataset_root = os.path.join(dataset_path, dataset_name)

        print("\n" + "=" * 60)
        print(f"Checking dataset: {dataset_name}")
        print("=" * 60)

        if not os.path.exists(dataset_root):
            print(f"Dataset folder was not found: {dataset_root}")
            continue

        print(f"\nContents of {dataset_name}:")
        print(os.listdir(dataset_root))

        for split in ["train_set", "eval_set"]:
            split_path = os.path.join(dataset_root, split)

            print(f"\n{dataset_name}/{split}:")

            if not os.path.exists(split_path):
                print(f"Split folder was not found: {split_path}")
                continue

            print("Contents:", os.listdir(split_path))

            for label in ["fake", "real"]:
                label_path = os.path.join(split_path, label)

                if not os.path.exists(label_path):
                    print(f"Label folder was not found: {label_path}")
                    continue

                files = [
                    f for f in os.listdir(label_path)
                    if f.lower().endswith((".wav", ".mp3"))
                ]

                print(f"{label}: {len(files)} files")