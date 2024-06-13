import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('https://github.com/DesireeDomingo-BSIT2B/image-classification/blob/main/save_model.keras')

def process_image(img):
    # Preprocess the image as needed (resize, normalization, etc.)
    img = img.resize((224, 224))  # Adjust size as per your model's input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    processed_image = process_image(image)
    prediction = model.predict(processed_image)
    return prediction

def main():
    st.title("Image Processing App")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict(image)
        st.write(prediction)

if __name__ == '__main__':
    main()
