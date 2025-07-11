from source.preprocessing import get_datasets
from source.model_training import build_model, train_model
from source.evaluation_utils import evaluate_model
from source.export_utils import export_model
from source.utils import load_config

def main():
    # Load configs
    train_cfg = load_config("config/training.yaml")
    model_cfg = load_config("config/model_params.yaml")
    infer_cfg = load_config("config/inference.yaml")

    img_size = (train_cfg["img_height"], train_cfg["img_width"])

    print("🔄 Loading data...")
    train_ds, val_ds, class_names = get_datasets(
        dataset_path=train_cfg["dataset_path"],
        img_size=img_size,
        batch_size=train_cfg["batch_size"],
        validation_split=train_cfg["validation_split"]
    )

    print("🧠 Building model...")
    model = build_model(
        num_classes=len(class_names),
        input_shape=img_size + (3,),
        dropout=model_cfg["dropout"],
        dense_units=model_cfg["dense_units"]
    )

    print("🚀 Training model...")
    model = train_model(
        model, train_ds, val_ds,
        log_dir=train_cfg["log_dir"],
        checkpoint_path=train_cfg["checkpoint_path"],
        epochs=train_cfg["epochs"]
    )

    print("📊 Evaluating model...")
    evaluate_model(model, val_ds, class_names, output_dir="experiments/metrics")

    print("💾 Exporting model...")
    export_model(model, h5_path=infer_cfg["model_path"], tflite_path=infer_cfg["tflite_path"])