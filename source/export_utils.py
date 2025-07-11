import tensorflow as tf

def export_model(model, tflite_path="model.tflite"):
    # Save to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved at {tflite_path}")
