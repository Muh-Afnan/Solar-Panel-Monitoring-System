# model_training.py
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_model(num_classes, input_shape=(256, 256, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    ) 
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, train_ds, val_ds, log_dir="logs", epochs=10):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks_list = [
        callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        callbacks.TensorBoard(log_dir=log_dir)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks_list)
    return model
