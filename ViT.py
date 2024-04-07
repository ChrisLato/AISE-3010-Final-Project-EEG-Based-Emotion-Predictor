import tensorflow as tf
from tensorflow.keras import layers, models

def ViTModel(feature_shape, num_classes):
    # Define the input layer with the shape of features extracted by the CNN.
    inputs = layers.Input(shape=feature_shape)

    # MultiHeadAttention layer performs self-attention, allowing the model to weigh the importance of different parts of the input.
    # 'num_heads' determines the number of attention heads. More heads allow the model to focus on different parts of the input simultaneously.
    # 'key_dim' specifies the size of each attention head for query and key.
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(inputs, inputs)

    # LayerNormalization normalizes the inputs across the features. It is essential for stabilizing the training of transformers.
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # GlobalAveragePooling1D reduces the dimensionality of the input by computing the average of each dimension, preparing it for the dense layer.
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layer with ReLU activation to introduce non-linearity, allowing for complex mappings between the input and output.
    x = layers.Dense(512, activation="relu")(x)

    # Dropout layer to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
    x = layers.Dropout(0.5)(x)

    # Output Dense layer with 'softmax' activation to classify the input into 'num_classes' categories.
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create the Model object, specifying inputs and outputs. This completes the model's architecture.
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model with the Adam optimizer and categorical crossentropy loss function. The accuracy metric is used for evaluation.
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

# Example usage:
feature_shape = (81, 128)  # This should match the output shape of your CNN after reshaping for ViT compatibility.
num_classes = 3  # Update based on your dataset. Represents the number of distinct categories to classify.
model = ViTModel(feature_shape, num_classes)
model.summary()  # Prints a summary of the model's architecture, showing the flow of data and parameters count.
