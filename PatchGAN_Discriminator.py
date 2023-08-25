from tensorflow import keras
import tensorflow as tf
from keras import layers, Model

class PatchGAN_Discriminator(Model):
    """Class for building and training a PatchGAN Discriminator model.
    
    The PatchGAN Discriminator is designed to classify whether each patch in an image is real or fake.
    """
    
    def __init__(self, image_size):
        """Initialize the PatchGAN_Discriminator.
        
        Args:
            image_size (int): The size of the input images.
        """
        super(PatchGAN_Discriminator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.image_size = image_size
        self.build(image_size)
    
    def build(self, input_shape):
        """Initialize the layers of the model."""
        self.input_layer = layers.InputLayer(input_shape=[self.image_size, self.image_size, 3])
        
        # Downsample layers
        self.down1 = self.downsample(64, 4)
        self.down2 = self.downsample(128, 4)
        self.down3 = self.downsample(256, 4)
        
        # Convolution layers
        self.conv1 = layers.Conv2D(256, (4, 4,), strides=(1, 1), padding='same', kernel_initializer=self.initializer)
        self.conv2 = layers.Conv2D(512, 4, strides=1, kernel_initializer=self.initializer, use_bias=False)
        self.conv3 = layers.Conv2D(1, 4, strides=1, kernel_initializer=self.initializer)
        
        # Normalization and activation layers
        self.norm_layer = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        
        # Padding layers
        self.zero_pad1 = layers.ZeroPadding2D()
        self.zero_pad2 = layers.ZeroPadding2D()
        
        # Final layer to flatten output
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation = 'sigmoid')
    
    def call(self, inputs):
        """Forward pass for the model.
        
        Args:
            input (tf.Tensor): The input tensor.
        
        Returns:
            tf.Tensor: The output tensor. The output will be the "patch-wise classification score," referring to the model's prediction of whether the image is real/fake.
            A higher patch-wise classification score indicates the discriminator believes the image is more likely to be real, and a lower score indicates the discriminator believes the image is more likely to be fake.
        """
        x = self.input_layer(inputs)

        #Downsample the images
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        #Apply the first convolutional layer
        x = self.conv1(x)

        #Apply padding
        x = self.zero_pad1(x)

        #Apply second convolutional layer
        x = self.conv2(x)

        #Apply normalization
        x = self.norm_layer(x)
        x = self.leaky_relu(x)
        x = self.zero_pad2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        last = self.dense(x)
        
        return last
    
    def downsample(self, num_filters, size, apply_batchnorm=True):
        """Downsample the image.
        
        Args:
            num_filters (int): Number of filters for the Conv2D layer.
            size (int): Kernel size for the Conv2D layer.
            apply_batchnorm (bool): Whether to apply batch normalization.
        
        Returns:
            keras.Sequential: A sequential model for downsampling.
        """
        initializer = tf.random_normal_initializer(0, 0.02)
        model = keras.Sequential()
        model.add(layers.Conv2D(num_filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
        
        return model
