"""Streamlit demo app skeleton for MNIST classifier
Run after training and saving mnist_cnn.h5: streamlit run streamlit_app.py
This app allows upload of an image and returns predicted digit.
"""
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.title('MNIST Classifier Demo')

model = None
try:
    model = tf.keras.models.load_model('mnist_cnn.h5')
except Exception as e:
    st.warning('Model file mnist_cnn.h5 not found. Run mnist_cnn.py to train and save the model.')

uploaded = st.file_uploader('Upload a 28x28 grayscale image of a handwritten digit (PNG/JPG)')
if uploaded is not None and model is not None:
    img = Image.open(uploaded).convert('L')
    img = ImageOps.fit(img, (28,28))
    st.image(img, caption='Input image', width=150)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    preds = model.predict(arr)
    pred = np.argmax(preds, axis=1)[0]
    st.write(f'Predicted digit: {pred}')
