"""Fixed TensorFlow example and notes for common TensorFlow bugs
This file demonstrates typical fixes: correct input shapes, correct loss for sparse labels, and final activation for multi-class.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Common buggy code patterns fixed here:
# - Using Dense(10) with no activation and using binary_crossentropy -> wrong loss for multi-class
# - Passing labels as one-hot but using sparse_categorical_crossentropy (or vice-versa)
# - Mismatched input shape (e.g., forgetting channel dimension)


def build_fixed_model():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # softmax for multi-class
    ])
    # Use sparse_categorical_crossentropy when labels are integers 0..9
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def quick_smoke_test():
    # Create tiny random data to validate shapes and training loop
    x = np.random.rand(16,28,28,1).astype('float32')
    y = np.random.randint(0,10,size=(16,))
    m = build_fixed_model()
    print(m.summary())
    m.train_on_batch(x,y)
    print('Smoke test training step completed')

if __name__ == '__main__':
    quick_smoke_test()
