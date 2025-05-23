import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# 1) Curvas de pérdida por fold
plt.figure()
for path in sorted(glob.glob("results/fold*_losses.csv")):
    df = pd.read_csv(path)
    fold = int(path.split("fold")[1].split("_")[0])
    plt.plot(df["epoch"], df["train_loss"], label=f"Train Fold {fold}")
    plt.plot(df["epoch"], df["val_loss"],   label=f" Val  Fold {fold}", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Curvas de pérdida por fold")
plt.tight_layout()
plt.savefig("results/loss_per_fold.png", dpi=150)
plt.close()

# 2) Curvas ROC individuales + promedio
all_preds = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/fold*_preds.csv"))],
                      ignore_index=True)
mean_fpr = np.linspace(0, 1, 100)
tpr_interp = []

plt.figure()
for fold, grp in all_preds.groupby("fold"):
    fpr, tpr, _ = roc_curve(grp["y_true"], grp["y_score"])
    plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {int(fold)}")
    tpr_interp.append(np.interp(mean_fpr, fpr, tpr))
mean_tpr = np.mean(tpr_interp, axis=0)
std_tpr  = np.std(tpr_interp, axis=0)
plt.plot(mean_fpr, mean_tpr, color="black", linewidth=2, label="ROC media")
plt.fill_between(mean_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr, alpha=0.2)
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curvas ROC por fold y promedio")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("results/roc_mean.png", dpi=150)
plt.close()

# 3) Curvas Precision–Recall individuales + promedio
mean_recall = np.linspace(0, 1, 100)
prec_interp = []

plt.figure()
for fold, grp in all_preds.groupby("fold"):
    prec, rec, _ = precision_recall_curve(grp["y_true"], grp["y_score"])
    plt.plot(rec, prec, alpha=0.3, label=f"Fold {int(fold)}")
    # invertimos para interpolar
    prec_interp.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
mean_prec = np.mean(prec_interp, axis=0)
std_prec  = np.std(prec_interp, axis=0)
plt.plot(mean_recall, mean_prec, color="black", linewidth=2, label="PR media")
plt.fill_between(mean_recall, mean_prec-std_prec, mean_prec+std_prec, alpha=0.2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curvas Precision–Recall por fold y promedio")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("results/pr_mean.png", dpi=150)
plt.close()
