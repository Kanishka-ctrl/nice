import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load your models
MODEL = tf.keras.models.load_model('./potato_trained_models/1/')
TOMATO_MODEL = tf.keras.models.load_model('./tomato_trained_models/1')
PEEPER_MODEL = tf.keras.models.load_model('./pepper_trained_models/1')

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
Tomato_classes = ['Tomato_healthy', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato_Septoria_leaf_spot',
 'Tomato__Tomato_mosaic_virus', 'Tomato_Leaf_Mold', 'Tomato_Bacterial_spot', 'Tomato_Late_blight',
 'Tomato_Early_blight', 'Tomato__Tomato_YellowLeaf__Curl_Virus']
pepper_classes = ['pepper_healthy', 'pepper_bell_bacterial_spot']

st.set_page_config(
    layout="wide",
    page_title='Plant Disease Detection',
)

st.title("Plant Disease Detection")
st.write("This application detects diseases in potatoes, tomatoes, and peppers.")
options = ["Select One Plant", "Tomato", "Potato", "Pepper"]

# Create a selectbox for the user to choose one option
selected_option = st.selectbox("Select Plant:", options)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def read_file_as_image(file):
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    return image

def predict_disease(image, model, class_names):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class, confidence

if uploaded_file is not None:
    image = read_file_as_image(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    if selected_option == 'Potato':
        predicted_class, confidence = predict_disease(image, MODEL, class_names)
        st.write("Predicted Class: ", predicted_class, "Confidence Level: ", confidence)
    elif selected_option == 'Tomato':
        predicted_class, confidence = predict_disease(image, TOMATO_MODEL, Tomato_classes)
        st.write("Predicted Class: ", predicted_class, "Confidence Level: ", confidence)
    elif selected_option == 'Pepper':
        predicted_class, confidence = predict_disease(image, PEEPER_MODEL, pepper_classes)
        st.write("Predicted Class: ", predicted_class, "Confidence Level: ", confidence)
    else:
        st.write("Please select a valid plant type.")
