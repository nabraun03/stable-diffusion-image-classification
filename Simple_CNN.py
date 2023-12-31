#Model inspired by Sahil Danayak on Kaggle
#Source: https://www.kaggle.com/code/sahildanayak/real-and-ai-images-classifier
#Copyright 2023 Sahil Danayak
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow import keras
import tensorflow as tf
from keras import layers, Model

class Simple_CNN(Model):
    """Class for building and training a PatchGAN Discriminator model.
    
    The PatchGAN Discriminator is designed to classify whether each patch in an image is real or fake.
    """
    
    def __init__(self, image_size):
        """Initialize the PatchGAN_Discriminator.
        
        Args:
            image_size (int): The size of the input images.
        """
        super(Simple_CNN, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.image_size = image_size
        self.build(image_size)
    
    def build(self, input_shape):
        """Initialize the layers of the model."""
        self.input_layer = layers.InputLayer(input_shape=[self.image_size, self.image_size, 3])
        
        self.conv1 = layers.Conv2D(32, (4, 4,), strides=(1, 1), padding='same', kernel_initializer=self.initializer)
        self.conv2 = layers.Conv2D(32, 4, strides = 1, kernel_initializer=self.initializer)

        self.max_pooling = layers.MaxPool2D(pool_size=(2, 2))

        self.dropout1 = layers.Dropout(0.25)
        self.dropout2 = layers.Dropout(0.5)

        self.flatten = layers.Flatten()

        self.dense1 = layers.Dense(256, activation = 'relu')
        self.dense2 = layers.Dense(1, activation = 'sigmoid')
    
    def call(self, inputs):
        """Forward pass for the model.
        
        Args:
            input (tf.Tensor): The input tensor.
        
        Returns:
            tf.Tensor: The output tensor. The output will be the "patch-wise classification score," referring to the model's prediction of whether the image is real/fake.
            A higher patch-wise classification score indicates the discriminator believes the image is more likely to be real, and a lower score indicates the discriminator believes the image is more likely to be fake.
        """
        x = self.input_layer(inputs)

        #Apply convolutional layers with max pooling between
        x = self.conv1(x)
        x = self.max_pooling(x)
        x = self.conv2(x)

        #Apply dropout before dense layers
        x = self.dropout1(x)

        #Flatten
        x = self.flatten(x)

        #Apply dense layer with dropout after
        x = self.dense1(x)
        x = self.dropout2(x)

        #Apply final dense layer with sigmoid activation
        x = self.dense2(x)

        return x
    
