import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs("image", exist_ok=True)

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

for i in range(10):
    image = x_train[i]
    image_pil = Image.fromarray(image)
    image_pil.save(f"image/image_{i}.png")
