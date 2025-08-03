import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50,EfficientNetB0
from tensorflow.keras import layers, models, callbacks,regularizers
from sklearn.utils import class_weight
import numpy as np

def build_model(name,num_classes, input_shape, dropout,regularizer):
    if name == 'mobilenet':
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        ) 
    elif name == 'vgg16':
            base_model = VGG16(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif name == 'resnet50':
            base_model = ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )

                
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(regularizer)),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, train_ds, val_ds, log_dir,checkpoint_path ,epochs):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks_list = [
        callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        callbacks.TensorBoard(log_dir=log_dir),
        lr_scheduler
    ]

    y_train = []

    for _, labels in train_ds:
        class_indices = tf.argmax(labels, axis=1)
        y_train.extend(class_indices.numpy())

    y_train = np.array(y_train)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks_list,class_weight=class_weight_dict)
    return model,history
