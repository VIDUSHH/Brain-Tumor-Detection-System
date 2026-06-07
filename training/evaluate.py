import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from training.data_loader import get_data_loaders

def evaluate_model():
    # Setup directories
    eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "evaluation"))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(eval_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "best_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.\n"
            f"Please run the training script first: python training/train.py"
        )

    print(f"[Evaluation] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("[Evaluation] Loading test dataset...")
    # Consistent image size of 224
    _, test_ds = get_data_loaders(image_size=224, batch_size=32)

    print("[Evaluation] Running inference on testing set...")
    # Get all class probability predictions
    all_probs = model.predict(test_ds, verbose=1)
    all_preds = np.argmax(all_probs, axis=1)

    # Extract true labels from the tf.data.Dataset
    true_labels_list = []
    for _, labels in test_ds:
        true_labels_list.append(labels.numpy())
    
    # Concatenate all batches of true labels
    all_labels_onehot = np.concatenate(true_labels_list, axis=0)
    all_labels_indices = np.argmax(all_labels_onehot, axis=1)

    class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    # 1. Classification Report (Precision, Recall, F1)
    report_dict = classification_report(
        all_labels_indices,
        all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    report_txt = classification_report(
        all_labels_indices,
        all_preds,
        target_names=class_names
    )
    
    print("\n=== Classification Report ===")
    print(report_txt)

    # 2. Multi-class ROC-AUC (One-vs-Rest)
    roc_auc_ovr = roc_auc_score(
        all_labels_onehot,
        all_probs,
        multi_class="ovr",
        average="macro"
    )
    print(f"ROC-AUC (One-vs-Rest, Macro-averaged): {roc_auc_ovr:.4f}")
    
    report_dict["roc_auc_macro_ovr"] = float(roc_auc_ovr)

    # Save metrics in JSON format
    metrics_json_path = os.path.join(eval_dir, "metrics_report.json")
    with open(metrics_json_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"[Evaluation] Saved JSON report to: {metrics_json_path}")

    # 3. Plot and Save Confusion Matrix
    cm = confusion_matrix(all_labels_indices, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        cbar=False,
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title("Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Labels", fontsize=11, labelpad=10)
    plt.ylabel("True Labels", fontsize=11, labelpad=10)
    plt.tight_layout()
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrix plot to: {cm_path}")

    # 4. Plot and Save ROC Curve (for each class)
    plt.figure(figsize=(9, 7))
    colors = ["#4A90E2", "#FF4B4B", "#2ECC71", "#9B59B6"]
    
    # Calculate ROC curve and ROC area for each class
    for idx, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(all_labels_onehot[:, idx], all_probs[:, idx])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=2.5,
            label=f"{class_name.capitalize()} (AUC = {roc_auc_val:.3f})"
        )

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(eval_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved ROC Curves plot to: {roc_path}")
    print("\n[Evaluation Complete] All reports and visualization files saved in: artifacts/evaluation/")

if __name__ == "__main__":
    evaluate_model()
