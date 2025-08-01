import os 
from tensorflow.keras.models import save_model

def export_model (model, path, model_name):
    
    # Create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the model in TensorFlow SavedModel format
    full_path = os.path.join(path, model_name + ".h5")  # or ".keras"
    model.save(full_path)  # .keras or .h5 can be used as well

    print(f"✅ Model saved successfully at: {full_path}")