import tensorflow as tf
from tensorflow.keras import layers, models

class ResBlock(layers.Layer):
    """
    A residual block that implements two convolutional layers.
    The first convolution uses ReLU activation, and the second uses no activation.
    An identity shortcut adds the block's input to its output, helping with gradient flow.
    """
    def __init__(self, out_channels, ksize, **kwargs):
        super().__init__(**kwargs)
        # First convolution increases the depth to 'out_channels' and applies ReLU.
        self.conv1 = layers.Conv2D(out_channels, (ksize, ksize), padding='same', activation='relu')
        # Second convolution with no activation function; maintains depth.
        self.conv2 = layers.Conv2D(out_channels, (ksize, ksize), padding='same')

    def call(self, inputs):
        # Apply the first convolution
        x = self.conv1(inputs)
        # Apply the second convolution
        x = self.conv2(x)
        # Add the input (identity shortcut) to the output of the convolutions
        return layers.add([x, inputs])

def build_cnn_for_vit(input_shape):
    """
    Builds a CNN designed for feature extraction to feed into a ViT model.
    The output is reshaped to match the ViT's expected input shape.
    """
    # Input layer specifying the shape of input images.
    inputs = layers.Input(shape=input_shape)
    
    # First convolution layer to process the input image and increase its depth.
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    
    # Apply several ResBlocks to capture features while maintaining dimensions.
    for _ in range(3):
        x = ResBlock(32, 3)(x)  # Each applies 32 filters of size 3x3
    
    # Increase the depth further while keeping spatial dimensions.
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    for _ in range(3):
        x = ResBlock(64, 3)(x)  # Each applies 64 filters of size 3x3
    
    # Final depth increase before reshaping for the ViT.
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    for _ in range(3):
        x = ResBlock(128, 3)(x)  # Each applies 128 filters of size 3x3
    
    # Reshape the CNN output to a sequence of vectors, each representing a "patch" for the ViT.
    # This prepares the features in a form that the ViT can process.
    x = layers.Reshape((81, 128))(x)
    
    # Define and return the model.
    model = models.Model(inputs=inputs, outputs=x)
    return model

# Define the shape of images your CNN will receive. This should match your dataset.
input_shape = (9, 9, 5)
# Build the model with the specified input shape.
model = build_cnn_for_vit(input_shape)
# Print a summary to see the model's structure.
model.summary()
