import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3, 3), activation="relu", input_shape =(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentrppy', metrics=['accuracy'])

print('Tensorflow CNN model ready to roll.....')

import torch.nn as nn
import torch.nn.functional as F

class SimpleCnn(nn.Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()

        #  Extracts features from the input image.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, activation='relu')
        # Downsamples feature maps to reduce computation and prevent overfitting
        self.pool = nn.max_pool2d(2, 2)
        # Performs classification based on extracted features
        self.fc1 = nn.Linear(32*15*15, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print(x)
        x = self.pool(x)
        x = x.view(-1, 32*15*15)
        x = F.relu(self.fc1(x))
        x= self.fc2(x)

print ("Pytorch CNN model ready.....")    
