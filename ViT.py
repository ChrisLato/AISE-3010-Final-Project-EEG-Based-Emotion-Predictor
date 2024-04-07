import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_addons as tfa
import CNN  # Importing the CNN model function

def vit_classifier(feature_shape, num_classes):
    inputs = layers.Input(shape=feature_shape)
    transformer_encoder = tfa.layers.MultiHeadAttention(head_size=128, num_heads=4)
    x = transformer_encoder(inputs, inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_for_vit(features, patch_size=(3, 3), num_patches=81):
    """
    Adjusts the CNN's output to be suitable for the ViT model.
    Assumes `features` is the output of the CNN model.
    """
    num_features = features.shape[-1]
    patch_features = patch_size[0] * patch_size[1] * num_features
    return np.reshape(features, (-1, num_patches, patch_features))