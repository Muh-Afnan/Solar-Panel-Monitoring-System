import yaml
import os
from datetime import datetime
import tensorflow as tf


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def create_experiment_folders(base_model,base_dir):
    timestamp = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(base_dir,base_model,timestamp)
    os.makedirs(exp_dir,exist_ok = True)
    sub_folders = {
        # "root": exp_dir,
        "logs":os.path.join(exp_dir,"logs"),
        "checkpoints":os.path.join(exp_dir,"checkpoints"),
        "metrics":os.path.join(exp_dir,"metrics")
    }
    for path in sub_folders.values():
        os.makedirs(path, exist_ok=True)
    return sub_folders

def remove_tensorflow_invalid_images(dataset_dir, allowed_exts={'.jpg', '.jpeg', '.png', '.bmp', '.gif'}):
    removed = 0

    def is_image_ok_tf(filepath):
        try:
            img_bytes = tf.io.read_file(filepath)
            tf.image.decode_image(img_bytes)
            return True
        except:
            return False

    for subdir, _, files in os.walk(dataset_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            file_path = os.path.join(subdir, file)

            if ext not in allowed_exts or not is_image_ok_tf(file_path):
                try:
                    os.remove(file_path)
                    print(f"❌ Removed invalid image: {file_path}")
                    removed += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {file_path}: {e}")

    print(f"✅ Removed {removed} TensorFlow-invalid image(s).")
