import tensorflow as tf
from tensorflow.keras import layers

def get_datasets(dataset_path, width, height,batch_size, validation_split):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels = "inferred",
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=123,
        label_mode="categorical"
    )
    
    class_names = train_ds.class_names

    train_ds = train_ds.map(lambda x, y: (tf.image.resize_with_pad(x, width, height), y))


    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels = "inferred",
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        label_mode="categorical"
    )
    val_ds = val_ds.map(lambda x, y: (tf.image.resize_with_pad(x,width ,height), y))


    # class_names = train_ds.class_names


    aug = tf.keras.Sequential([
        layers.Rescaling(1./255),
        # layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        # layers.RandomBrightness(0.2),
    ])

    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y))
    val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE), class_names
    # return train_ds, val_ds, class_names
