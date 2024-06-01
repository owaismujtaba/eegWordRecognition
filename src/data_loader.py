import tensorflow as tf
import config

class ImageDataLoader:
    def __init__(self, data_dir, img_size=(config.imageSize, config.imageSize), batch_size=config.batchSize, validation_split=0.1):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.train_ds, self.val_ds = self.load_datasets()

    def load_datasets(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )

        return train_ds, val_ds

    def get_train_dataset(self):
        return self.train_ds

    def get_val_dataset(self):
        return self.val_ds

    def get_class_names(self):
        return self.train_ds.class_names

    @staticmethod
    def preprocess_dataset(dataset):
        dataset = dataset.map(convert_to_grayscale)
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
        ])

        return dataset.map(lambda x, y: (data_augmentation(x, training=True), y))


def convert_to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def getDataLoaders():
    data_dir = config.imageDataDirectory
    data_loader = ImageDataLoader(data_dir)

    train_ds = data_loader.get_train_dataset()
    val_ds = data_loader.get_val_dataset()

    train_ds = ImageDataLoader.preprocess_dataset(train_ds)
    val_ds = ImageDataLoader.preprocess_dataset(val_ds)

    return train_ds, val_ds


