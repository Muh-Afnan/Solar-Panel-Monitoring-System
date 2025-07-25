import yaml
import os
from datetime import datetime
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image

from PIL import Image, UnidentifiedImageError


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def create_experiment_folders(base_dir="experiments"):
    timestamp = datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(base_dir,timestamp)

    sub_folders = {
        "root": exp_dir,
        "logs":os.path.join(exp_dir,"logs"),
        "checkpoints":os.path.join(exp_dir,"logs"),
        "metrics":os.path.join(exp_dir,"logs")
    }
    for path in sub_folders:
        os.makedirs(path, exist_ok=True)
    return sub_folders

import os
import tensorflow as tf

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
