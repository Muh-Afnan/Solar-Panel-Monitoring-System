import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

def evaluate_model(model, val_ds, class_names, output_dir):
    y_true, y_pred = [], []

    for images, labels in val_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    # Evaluate model (Keras style)
    results = model.evaluate(val_ds, verbose=0)
    metrics = {"loss": results[0], "accuracy": results[1]}
    # Save metrics to JSON
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Optional print (for CLI feedback)
    print("\n📊 Evaluation Results:")
    print(json.dumps(metrics, indent=4))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ✅ Return useful data
    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm
    }
