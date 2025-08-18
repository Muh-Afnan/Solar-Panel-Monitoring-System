import tensorflow as tf
from tensorflow.keras.models import load_model
from source.utils import load_config
import os


def preprocess_img(img,width,height):
    ready_img = tf.image.resize_with_pad(img, width, height)
    return ready_img

def predict(img,model_path):
    model = load_model(model_path)
    result = model.predict()



def main():
    train_cfg = load_config("config/training.yaml")
    infer_cfg = load_config("config/inference.yaml")
    path = infer_cfg["model_path"]
    model = infer_cfg["model"]
    model_path = os.path.join(path,model)
    width = train_cfg["img_width"]
    height = train_cfg["img_height"]

    ready_img = preprocess_img("img",width,height)

    predict(ready_img,model_path)

    



