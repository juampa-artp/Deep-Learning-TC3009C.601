import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

COLORS = plt.get_cmap("tab10").colors

def plot_loss(train, val):
    epochs_range = range(1, len(train) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train, color="blue", linewidth=2, label="Train Loss")
    plt.plot(epochs_range, val, color="orange", linewidth=2, label="val Loss")

    plt.title("Evolución de la Pérdida de Entrenamiento contra Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (Loss)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def conf_mat(cm, classes, state=0):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.xticks(np.arange(len(classes)), classes, rotation=90)
    plt.yticks(np.arange(len(classes)), classes)

    if state == 0:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()

def conf_mat_norm(cm, classes, state=1):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    conf_mat(cm_norm, classes, state)


def target_histogram(target, classes, title="Distribucion del target"):
    target = np.asarray(target)
    classes = list(classes)

    # Usar la misma posición para cada label y su barra.
    mapping = {label: index for index, label in enumerate(classes)}
    target = np.array([mapping[label] for label in target])

    bins = np.arange(len(classes) + 1) - 0.5

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(target, bins=bins, color="#2878b5", edgecolor="white", rwidth=0.86)

    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("Clase")
    ax.set_ylabel("Numero de observaciones")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")

    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()


def roc_curve_plot(model, X_test, y_test, classes):
    y_score = model.predict_proba(X_test)
    n_classes = len(classes)
    y_onehot_test = label_binarize(y_test, classes=classes)

    fpr, tpr, roc_auc = dict(), dict(), dict()

    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Micro AUC: {roc_auc['micro']:.2f}")
    print(f"Macro AUC: {roc_auc['macro']:.2f}")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr["micro"], tpr["micro"],
            label=f"Micro-average (AUC = {roc_auc['micro']:.2f})",
            color="deeppink", linestyle=":", linewidth=3)

    ax.plot(fpr["macro"], tpr["macro"],
            label=f"Macro-average (AUC = {roc_auc['macro']:.2f})",
            color="navy", linestyle=":", linewidth=3)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, color in enumerate(colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})"
        )

    ax.plot([0, 1], [0, 1], "--", color="gray")

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve - One-vs-Rest"
    )

    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fpr, tpr, roc_auc

def load_learning_curve(path):
    data = joblib.load(path)

    train_sizes = data["train_sizes"]
    train_scores = data["train_scores"]
    val_scores = data["val_scores"]

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))

    plt.plot(train_sizes, train_mean, "o-", label="Training")
    plt.plot(train_sizes, val_mean, "o-", label="Validation")

    plt.xlabel("Training examples")
    plt.ylabel("F1 Macro")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()