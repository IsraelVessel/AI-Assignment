"""MNIST CNN training script (TensorFlow Keras)
Run: python mnist_cnn.py
Note: For >95% accuracy, run for more epochs (8-12) and use GPU (Colab recommended)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = build_model()
    model.summary()

    # Train
    history = model.fit(x_train, y_train, epochs=6, batch_size=128, validation_split=0.1)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', test_acc)

    # Save model
    model.save('mnist_cnn.h5')


if __name__ == '__main__':
    main()
