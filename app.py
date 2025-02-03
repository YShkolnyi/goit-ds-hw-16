import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.title("Розпізнавання одягу")

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

selected_model = st.sidebar.selectbox(
    "Оберіть модель для передбачення:",
    ["Модель CNN", "Модель VGG16"]
)

if selected_model == "Модель CNN":
    model = tf.keras.models.load_model('CNN.keras')
elif selected_model == "Модель VGG16":
    model = tf.keras.models.load_model('VGG16.keras')

def preprocess_image(image, model_type):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    
    if model_type == "Модель VGG16":
        image = np.stack([image] * 3, axis=-1)
        image = tf.image.resize(image, (32, 32))
    
    return image

def predict_image(image, model_type):
    image = preprocess_image(image, model_type)
    image_exp = np.expand_dims(image, axis=0)
    if model_type == "Модель CNN":
        image_exp = np.expand_dims(image_exp, axis=-1)
    
    pred = model.predict(image_exp)
    probabilities = pred[0]
    predicted_class = np.argmax(probabilities)
    
    return predicted_class, probabilities

uploaded_file = st.file_uploader("Завантажте зображення:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Завантажене зображення", use_container_width=True)
    
    predicted_class, probabilities = predict_image(image, selected_model)
    st.write(f"Передбачений клас: {class_names[predicted_class]}")
    
    N = len(class_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    probabilities = probabilities.tolist()  

    probabilities += probabilities[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

    ax.plot(angles, probabilities, linewidth=3, linestyle='solid')
    ax.fill(angles, probabilities, alpha=0.5)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names)
    ax.set_title("Радіальна діаграма ймовірностей за класами", pad=20)

    st.pyplot(fig)
