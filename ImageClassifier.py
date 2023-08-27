import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import datetime
import zipfile

class ImageClassifier:
    """Class for building and training an image classifier for detecting real and AI-generated images."""
    
    # Constants for configuration
    FOLDER_PATH = 'C:/Users/nabra/Coding/Kaggle_competitions/AI_Detection/archive'
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    EPOCHS = 25
    BATCH_SIZE = 128
    
    def __init__(self):
        """Initialize the ImageClassifier and load training and test data."""
        self.train_data = self.load_data('/train')
        self.test_data = self.load_data('/test')
    
    @staticmethod
    def normalize(image, label):
        """Normalize the image data to the range [0, 1]."""
        return tf.divide(image, 255.0), label
    
    def load_data(self, data_type):
        """Load and preprocess data from a specified folder.
        
        Args:
            data_type (str): Type of data to load ('/train' or '/test').
        
        Returns:
            tf.data.Dataset: Preprocessed data.
        """
        try:
            data = keras.utils.image_dataset_from_directory(
                self.FOLDER_PATH + data_type, 
                image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH), 
                batch_size=self.BATCH_SIZE
            )
            data = data.map(self.normalize).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            print(f"{data_type} data loaded!")
            return data
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def compile_and_fit(self, model):
        """Compile and fit the model using the loaded data.
        
        Args:
            model (tf.keras.Model): The model to compile and fit.
        """
        model.compile(
            loss=keras.losses.BinaryCrossentropy(), 
            optimizer=keras.optimizers.Adam(), 
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Checkpoint callback to save model weights
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
        
        # TensorBoard callback
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

        # Fit the model
        model.fit(
            self.train_data, 
            epochs=self.EPOCHS, 
            validation_data=self.test_data, 
            shuffle=True, 
            verbose=1,
            callbacks=[checkpoint_callback, tensorboard_callback]
        )
