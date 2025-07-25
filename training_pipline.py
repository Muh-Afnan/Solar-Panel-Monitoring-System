from source.preprocessing import get_datasets
from source.model_training import build_model, train_model
from source.evaluation_utils import evaluate_model
from source.export_utils import export_model
from source.utils import load_config, create_experiment_folders
from source.utils import remove_tensorflow_invalid_images
import os

def main():

    exp_paths = create_experiment_folders()
    train_cfg = load_config("config/training.yaml")
    model_cfg = load_config("config/model_params.yaml")
    infer_cfg = load_config("config/inference.yaml")

    img_size = (train_cfg["img_height"], train_cfg["img_width"])
    remove_tensorflow_invalid_images("data/raw")


    print("🔄 Loading data...")
    train_ds, val_ds, class_names = get_datasets(
        dataset_path=train_cfg["dataset_path"],
        img_size=img_size,
        batch_size=train_cfg["batch_size"],
        validation_split=train_cfg["validation_split"]
    )

    best_accuracy = 0
    best_results = {}
    best_model = None

    print("🧠 Building model...")
    for base_model in model_cfg['base_model']:
        model = build_model(
            base_model,
            num_classes=len(class_names),
            input_shape=img_size + (3,),
            dropout=model_cfg["dropout"],
            # dense_units=model_cfg["dense_units"]
        )

        print("🚀 Training model...")
        model = train_model(
            model, train_ds, val_ds,
            log_dir=exp_paths["logs"],
            checkpoint_path = os.path.join(exp_paths["checkpoints"], "best_model.keras"),
            epochs=train_cfg["epochs"]
        )

        print("📊 Evaluating model...")
        results = evaluate_model(model, val_ds, class_names, output_dir="experiments/exp_xxx/metrics")
        print("Validation Accuracy:", results["metrics"]["accuracy"])
        acccuracy = results["metrics"]["accuracy"]
        if acccuracy >best_accuracy:
            best_accuracy = acccuracy
            best_results = results
            best_model = base_model

    print("💾 Exporting model...")
    export_model(model, h5_path=infer_cfg["model_path"], tflite_path=infer_cfg["tflite_path"])

if __name__ == "__main__":
    main()