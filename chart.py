import matplotlib.pyplot as plt

# =========================================================
# TRAINING HISTORY PLOT: WAVLM VS HUBERT
# =========================================================
# Purpose:
# - Compare F1-score curves between WavLM and HuBERT.
# - Use actual training logs only.
# - Show Train F1 and Eval F1 over available epochs.
# - Mark the best epoch for each model.
#
# Notes:
# - WavLM training was interrupted after Epoch 4.
# - HuBERT training log is available from Epoch 1 to Epoch 4.
# =========================================================

# =========================================================
# ACTUAL WAVLM RESULTS
# =========================================================
wavlm_epochs = [1, 2, 3, 4]

wavlm_train_f1 = [
    0.9738,
    0.9909,
    0.9944,
    0.9957,
]

wavlm_eval_f1 = [
    0.9839,
    0.9843,
    0.9800,
    0.9830,
]

# =========================================================
# ACTUAL HUBERT RESULTS
# =========================================================
hubert_epochs = [1, 2, 3, 4]

hubert_train_f1 = [
    0.9712,
    0.9897,
    0.9935,
    0.9951,
]

hubert_eval_f1 = [
    0.9817,
    0.9755,
    0.9956,
    0.9953,
]

# =========================================================
# FIND BEST EPOCHS BASED ON EVAL F1
# =========================================================
best_wavlm_idx = max(range(len(wavlm_eval_f1)), key=lambda i: wavlm_eval_f1[i])
best_hubert_idx = max(range(len(hubert_eval_f1)), key=lambda i: hubert_eval_f1[i])

best_wavlm_epoch = wavlm_epochs[best_wavlm_idx]
best_hubert_epoch = hubert_epochs[best_hubert_idx]

best_wavlm_f1 = wavlm_eval_f1[best_wavlm_idx]
best_hubert_f1 = hubert_eval_f1[best_hubert_idx]

# WavLM session was interrupted after epoch 4
wavlm_interrupted_epoch = 4

# HuBERT reached early stopping counter 1/3 at epoch 4
hubert_early_stopping_counter_epoch = 4

# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(16, 9))

# WavLM curves
plt.plot(
    wavlm_epochs,
    wavlm_train_f1,
    label="WavLM Train F1",
    color="tab:blue",
    linestyle="-",
    marker="o",
    linewidth=2.5,
    markersize=10,
)

plt.plot(
    wavlm_epochs,
    wavlm_eval_f1,
    label="WavLM Eval F1",
    color="tab:blue",
    linestyle="--",
    marker="s",
    linewidth=2.5,
    markersize=10,
)

# HuBERT curves
plt.plot(
    hubert_epochs,
    hubert_train_f1,
    label="HuBERT Train F1",
    color="tab:red",
    linestyle="-",
    marker="o",
    linewidth=2.5,
    markersize=10,
)

plt.plot(
    hubert_epochs,
    hubert_eval_f1,
    label="HuBERT Eval F1",
    color="tab:red",
    linestyle="--",
    marker="s",
    linewidth=2.5,
    markersize=10,
)

# =========================================================
# SHOW EVAL F1 VALUES
# =========================================================
for i, value in enumerate(wavlm_eval_f1):
    plt.text(
        wavlm_epochs[i],
        value + 0.0015,
        f"{value:.4f}",
        color="tab:blue",
        ha="center",
        va="bottom",
        fontsize=10,
    )

for i, value in enumerate(hubert_eval_f1):
    plt.text(
        hubert_epochs[i],
        value - 0.0025,
        f"{value:.4f}",
        color="tab:red",
        ha="center",
        va="top",
        fontsize=10,
    )

# =========================================================
# MARK BEST WAVLM EPOCH
# =========================================================
plt.axvline(
    x=best_wavlm_epoch,
    color="forestgreen",
    linestyle="--",
    linewidth=2.5,
)

plt.text(
    best_wavlm_epoch + 0.05,
    0.987,
    f"Best WavLM\nEpoch {best_wavlm_epoch}\nEval F1 = {best_wavlm_f1:.4f}",
    color="forestgreen",
    fontsize=12,
    fontweight="bold",
)

# =========================================================
# MARK BEST HUBERT EPOCH
# =========================================================
plt.axvline(
    x=best_hubert_epoch,
    color="purple",
    linestyle="--",
    linewidth=2.5,
)

plt.text(
    best_hubert_epoch + 0.05,
    0.992,
    f"Best HuBERT\nEpoch {best_hubert_epoch}\nEval F1 = {best_hubert_f1:.4f}",
    color="purple",
    fontsize=12,
    fontweight="bold",
)

# =========================================================
# MARK WAVLM INTERRUPTION
# =========================================================
plt.axvline(
    x=wavlm_interrupted_epoch,
    color="tab:gray",
    linestyle=":",
    linewidth=2.5,
)

plt.text(
    wavlm_interrupted_epoch + 0.05,
    0.970,
    "WavLM session\ninterrupted after\nEpoch 4",
    color="tab:gray",
    fontsize=11,
    fontweight="bold",
)

# =========================================================
# CHART STYLING
# =========================================================
plt.title(
    "Training History on Merged Dataset: WavLM-base vs HuBERT-base",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

plt.xlabel("Epoch", fontsize=14, fontweight="bold")
plt.ylabel("F1-Score", fontsize=14, fontweight="bold")

plt.xticks([1, 2, 3, 4])
plt.ylim(0.965, 1.002)

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right", fontsize=12)

plt.tight_layout()

plt.savefig(
    "training_history_merged_wavlm_vs_hubert.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()