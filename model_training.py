import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
from keras.utils import image_dataset_from_directory
from keras import Input, layers, Model

# Defining the directories for train, test, and validation datasets
train_dir = 'Spectr_dif/train'
test_dir = 'Spectr_dif/test'
val_dir = 'Spectr_dif/val'

# Preparing the image datasets
train_dataset = image_dataset_from_directory(train_dir, image_size=(360,360), batch_size=32)
validation_dataset = image_dataset_from_directory(test_dir, image_size=(360,360), batch_size=32)
test_dataset = image_dataset_from_directory(val_dir, image_size=(360,360), batch_size=32)

# Defining the model architecture
inputs = Input(shape=(360,360,3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)                
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)                
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)

# Compiling the model
model.compile(optimizer='adam', loss='mse', steps_per_execution=10, metrics=['accuracy'])

# Training the model
history = model.fit(train_dataset, epochs=25, validation_data=validation_dataset)

# Evaluating the model
scores = model.evaluate(test_dataset, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
