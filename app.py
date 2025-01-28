import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# Hide Streamlit's menu and footer
hide_streamlit_style = """
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Title for the app
st.markdown("""
    <h1 style='text-align: center; color: #FF6347;'>Tomato Leaf Disease Prediction</h1>
    <p style='text-align: center; font-size: 16px;'>Upload or capture a tomato leaf image to predict the disease.</p>
""", unsafe_allow_html=True)


def main():
    # File uploader section
    st.markdown('<h2 style="text-align: center; color: #FF6347;">Upload or Capture Tomato Leaf Image</h2>', unsafe_allow_html=True)
    
    # Option to use camera input
    file_uploaded = st.camera_input("Take a picture with your camera")

    if file_uploaded is not None:
        # Open and display the captured image
        image = Image.open(file_uploaded)
        st.image(image, caption='Captured Image', use_column_width=True)

        # Predict the disease class
        result, confidence, description, actions = predict_class(image)

        # Display results
        st.markdown(f"""
            <h3 style="color: #FF6347;">Prediction Result</h3>
            <p style="font-size: 18px;">The system has detected that the plant is infected with: <strong>{result}</strong></p>
            <p style="font-size: 18px;">Level of Accuracy: <strong>{confidence}%</strong></p>
            <h4 style="color: #FF6347;">Description:</h4>
            <p style="font-size: 16px;">{description}</p>
            <h4 style="color: #FF6347;">Recommended Actions:</h4>
            <p style="font-size: 16px;">{actions}</p>
        """, unsafe_allow_html=True)


def predict_class(image):
    with st.spinner('Loading Model...'):
        # Load the pre-trained model
        classifier_model = keras.models.load_model('tomatoes.h5', compile=False)
    
    # Resize the image for the model input
    test_image = image.resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0  # Normalize image
    test_image = np.expand_dims(test_image, axis=0)

    # Class names for the predictions
    class_name = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
                  'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
                  'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot', 
                  'Tomato_Tomato_YellowLeaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']

    # Class descriptions and actions
    descriptions = {
        'Tomato_Bacterial_spot': "A bacterial infection causing angular, water-soaked lesions.",
        'Tomato_Early_blight': "Early blight causes dark, concentric lesions on leaves and stems.",
        'Tomato_Late_blight': "Late blight causes dark, irregular spots on leaves, stems, and fruit.",
        'Tomato_Leaf_Mold': "Leaf mold causes yellowing of leaves with fuzzy mold growth on the underside.",
        'Tomato_Septoria_leaf_spot': "Septoria leaf spot causes small, circular, dark lesions on leaves.",
        'Tomato_Spider_mites_Two_spotted_spider_mite': "Spider mites create tiny yellow or brown spots on leaves.",
        'Tomato_Target_Spot': "Target spot causes dark, concentric circles in lesions on the leaves.",
        'Tomato_Tomato_YellowLeaf_Curl_Virus': "This virus causes leaf curling and yellowing in tomato plants.",
        'Tomato_Tomato_mosaic_virus': "Mosaic virus causes distorted, mottled leaves and stunted growth.",
        'Tomato_healthy': "The plant is healthy with no signs of disease."
    }
    
    actions = {
        'Tomato_Bacterial_spot': "Remove affected leaves and ensure proper irrigation to reduce water on foliage.",
        'Tomato_Early_blight': "Use fungicides and practice crop rotation to avoid early blight.",
        'Tomato_Late_blight': "Use resistant varieties and apply fungicides as soon as symptoms are seen.",
        'Tomato_Leaf_Mold': "Improve air circulation and remove affected leaves.",
        'Tomato_Septoria_leaf_spot': "Remove infected leaves and apply fungicides to prevent further spread.",
        'Tomato_Spider_mites_Two_spotted_spider_mite': "Use miticides and increase humidity to deter mite infestation.",
        'Tomato_Target_Spot': "Remove infected leaves and apply fungicides to control the spread.",
        'Tomato_Tomato_YellowLeaf_Curl_Virus': "Remove infected plants to prevent virus spread to healthy plants.",
        'Tomato_Tomato_mosaic_virus': "Remove and destroy infected plants immediately to prevent further spread.",
        'Tomato_healthy': "The plant is in good condition, continue with standard care."
    }

    # Predict and get the confidence
    prediction = classifier_model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]

    # Get description and recommended actions for the predicted disease
    description = descriptions[final_pred]
    action = actions[final_pred]

    return final_pred, confidence, description, action


# Main app execution
if __name__ == '__main__':
    main()
