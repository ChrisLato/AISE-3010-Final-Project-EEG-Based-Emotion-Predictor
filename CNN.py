import tensorflow as tf
from tensorflow.keras import layers, models

class ResBlock(layers.Layer):
    """
    A residual block that uses two convolutional layers and an identity shortcut.
    """
    def __init__(self, out_channels, ksize, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channels, (ksize, ksize), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(out_channels, (ksize, ksize), padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return layers.add([x, inputs])

def build_cnn_for_vit(input_shape):
    """
    Builds a CNN tailored for feature extraction for integration with a ViT model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolutional layer
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    
    # Adding residual blocks
    for _ in range(3):
        x = ResBlock(32, 3)(x)
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    for _ in range(3):
        x = ResBlock(64, 3)(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    for _ in range(3):
        x = ResBlock(128, 3)(x)
    
    # Flattening the output to make it suitable for ViT input.
    # Prepares the CNN output as a flat vector for each image.
    x = layers.Flatten()(x)
    
    # The output layer for specific task classification will be part of the ViT model.
    model = models.Model(inputs=inputs, outputs=x)
    
    return model

# Define the input shape of the images
input_shape = (9, 9, 5)  
model = build_cnn_for_vit(input_shape)
model.summary()
