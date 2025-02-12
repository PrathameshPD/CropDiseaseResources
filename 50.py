import os
import tensorflow as tf
import json
from PIL import Image
import numpy as np
import streamlit as st
import base64 
from googletrans import Translator
import cv2

st.set_page_config(layout='centered')

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "best_DenseNet121.keras")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
class_indices = json.load(open(os.path.join(working_dir, "class_indices.json")))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

def set_background(png_file):
    with open(png_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('imge1.png')

# Define recommendations based on disease and severity
recommendations = {
    "Brown spot": {
        "0": "No Disease Found, No action needed, monitor regularly.",
        "1": "No Spray.",
        "2": "No Spray.",
        "3": "No Spray.",
        "4": "Mancozeb 50% WP @ 2g/litre of water ratio.",
        "5": "Mancozeb 50% WP @ 2g/litre of water ratio.",
        "6": "Edifenophos 2ml / litre of water ratio.",
        "7": "Edifenophos 2ml / litre of water ratio.",
        "8": "Edifenophos 2ml / litre of water ratio.",
        "9": "Edifenophos 2ml / litre of water ratio."
    },
    "Leaf blast": {
        "0": "No Disease Found, No action needed, monitor regularly.",
        "1": "No Spray.",
        "2": "No Spray.",
        "3": "No Spray.",
        "4": "No Spray.",
        "5": "Spray the field with Carbendazim 50% WP 1 gm/l or  Tricyclazole 75%  WP 0.6gm/l.",
        "6": "Tebuconazole 50% + Trifloxystrobin 25% @ 0.4g per litre of water.",
        "7": "Tebuconazole 50% + Trifloxystrobin 25% @ 0.4g per litre of water.",
        "8": "Tebuconazole 50% + Trifloxystrobin 25% @ 0.4g per litre of water.",
        "9": "Tebuconazole 50% + Trifloxystrobin 25% @ 0.4g per litre of water."
    },
    "Narrow brown spot": {
        "0": "No Disease Found, No action needed, monitor regularly.",
        "1": "No Spray.",
        "2": "No Spray.",
        "3": "No Spray.",
        "4": "Mancozeb 50% WP @ 2g/litre of water ratio.",
        "5": "Mancozeb 50% WP @ 2g/litre of water ratio.",
        "6": "Edifenophos 2ml / litre of water ratio.",
        "7": "Edifenophos 2ml / litre of water ratio.",
        "8": "Edifenophos 2ml / litre of water ratio.",
        "9": "Edifenophos 2ml / litre of water ratio."
    },
    "Neck blast": {
        "0": "No Disease Found, No action needed, monitor regularly.",
        "1": "No Spray.",
        "3": "Spray the field with Carbendazim 50% WP 1 gm/l or  Tricyclazole 75%  WP 0.6gm/litre of water ratio.",
        "5": "Tebuconazole 50% + Trifloxystrobin 25% @ 0.4g per litre of water ratio.",
        "7": "Tebuconazole 50% + Trifloxystrobin 25% @ 0.4g per litre of water ratio.",
        "9": "Tebuconazole 50% + Trifloxystrobin 25% @ 0.4g per litre of water ratio."
    },
    "Sheath blight": {
        "0": "No Disease Found, No action needed, monitor regularly.",
        "1": "No Spray.",
        "3": "No Spray.",
        "5": "Spray the field with Carbendazim 50% WP 1 gm/l  or  Hexaconazole 5%  SC 2 ml/l.",
        "7": "Propiconazole 25%, EC 1 ml/l  or Thifluzamide  24 %  SC 1 ml/l.",
        "9": "Tricyclazole 45 % + Hexaconazole  10 % @  WG 1gm/l."
    },
    "Sheath rot": {
        "0": "No Disease Found, No action needed, monitor regularly.",
        "1": "No Spray.",
        "3": "No Spray.",
        "5": "Seed treatment with Captan or Thiram or Carbendazim or Carboxin or Tricyclazole @ 2g/kg.",
        "7": "Spray with Carbendazim 50 % WP 1g/l.",
        "9": "Spray with Carbendazim 50 % WP 1g/l."
    }
}

# Function to calculate infected percentage using OpenCV
def calculate_infected_percentage(image):
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    
    # Convert the image to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)

    # Define range for infected region color (you can adjust these values)
    lower_infected = np.array([10, 30, 30])  # Adjust these HSV values
    upper_infected = np.array([30, 255, 255])

    # Create a mask for infected areas
    mask = cv2.inRange(hsv, lower_infected, upper_infected)

    # Perform morphological operations to remove small noises in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Calculate the percentage of infected area
    infected_area = np.sum(mask_cleaned > 0)
    total_area = mask_cleaned.shape[0] * mask_cleaned.shape[1]
    infected_percentage = (infected_area / total_area) * 100

    return infected_percentage, mask_cleaned

# Function to estimate severity using specific scales
def estimate_severity(disease, area_percentage):
    if disease == "Leaf blast":
        if area_percentage < 0:
            return "0"
        elif area_percentage < 0.1:
            return "1"
        elif area_percentage < 0.3:
            return "2"
        elif area_percentage < 0.5:
            return "3"
        elif area_percentage < 2:
            return "4"
        elif area_percentage < 10:
            return "5"
        elif area_percentage < 25:
            return "6"
        elif area_percentage < 50:
            return "7"
        elif area_percentage < 75:
            return "8"
        else:
            return "9"
    elif disease == "Brown spot":
        if area_percentage < 0:
            return "0"
        elif area_percentage < 1:
            return "1"
        elif area_percentage < 3:
            return "2"
        elif area_percentage < 5:
            return "3"
        elif area_percentage < 10:
            return "4"
        elif area_percentage < 15:
            return "5"
        elif area_percentage < 25:
            return "6"
        elif area_percentage < 50:
            return "7"
        elif area_percentage < 75:
            return "8"
        else:
            return "9"      
    elif disease == "Narrow brown spot":
        if area_percentage < 0:
            return "0"
        elif area_percentage < 1:
            return "1"
        elif area_percentage < 3:
            return "2"
        elif area_percentage < 5:
            return "3"
        elif area_percentage < 10:
            return "4"
        elif area_percentage < 15:
            return "5"
        elif area_percentage < 25:
            return "6"
        elif area_percentage < 50:
            return "7"
        elif area_percentage < 75:
            return "8"
        else:
            return "9"
    elif disease == "Neck blast":
        if area_percentage < 0:
            return "0"
        elif area_percentage < 5:
            return "1"
        elif area_percentage < 10:
            return "3"
        elif area_percentage < 25:
            return "5"
        elif area_percentage < 50:
            return "7"
        else:
            return "9"
    elif disease == "Sheath blight":
        if area_percentage < 0:
            return "0"
        elif area_percentage < 20:
            return "1"
        elif area_percentage < 30:
            return "3"
        elif area_percentage < 45:
            return "5"
        elif area_percentage < 65:
            return "7"
        else:
            return "9"        
    elif disease == "Sheath rot":
        if area_percentage < 0:
            return "0"
        elif area_percentage < 1:
            return "1"
        elif area_percentage < 5:
            return "3"
        elif area_percentage < 25:
            return "5"
        elif area_percentage < 50:
            return "7"
        else:
            return "9"
        
def get_severity_category(severity_score):
    severity_score = int(severity_score)
    if severity_score <= 3:
        return "Low"
    elif 4 <= severity_score <= 5:
        return "Moderate"
    elif 6 <= severity_score <= 7:
        return "High"
    elif severity_score >= 8:
        return "Extreme"

# Function to estimate yield loss and growth based on disease and severity
def estimate_yield_growth_and_loss(disease, severity_score):
    severity_score = int(severity_score)
    
    if disease == "Leaf blast":
        yield_loss_mapping = {
            "5": 10,
            "6": 13,
            "7": 32,
            "8": 46,
            "9": 58
        }
    elif disease == "Neck blast":
        yield_loss_mapping = {
            "3": 10,
            "5": 16,
            "7": 38,
            "9": 58
        }
    elif disease == "Brown spot":
        yield_loss_mapping = {
            "4": 8,
            "5": 9,
            "6": 18,
            "7": 32,
            "8": 52,
            "9": 74
        }
    elif disease == "Narrow brown spot":
        yield_loss_mapping = {
            "4": 10,
            "5": 12,
            "6": 24,
            "7": 36,
            "8": 58,
            "9": 79
        }
    elif disease == "Sheath blight":
        yield_loss_mapping = {
            "5": 22,
            "7": 28,
            "9": 35
        }
    elif disease == "Sheath rot":
        yield_loss_mapping = {
            "5": 12,
            "7": 35,
            "9": 62
        }
    else:
        yield_loss_mapping = {}
    
    # Get yield loss if available
    yield_loss = yield_loss_mapping.get(str(severity_score))
    
    if yield_loss is not None:
        # Calculate yield growth as (100% - yield_loss)
        yield_growth = 100 - yield_loss
        # Display yield growth first, followed by yield loss
        return f"Yield Growth: {yield_growth}% | Yield Loss: {yield_loss}%"
    else:
        return "No yield loss and growth data available"

# Initialize the translator
translator = Translator()

# Function to translate text
def translate_text(text, dest_language):
    translation = translator.translate(text, dest=dest_language)
    return translation.text

# Function to display text based on selected language
def display_text(text, language):
    if language == "Kannada":
        return translate_text(text, 'kn')
    elif language == "Telugu":
        return translate_text(text, 'te')
    elif language == "Tamil":
        return translate_text(text, 'ta')
    elif language == "Hindi":
        return translate_text(text, 'hi')
    return text

# Sidebar for language selection
language = st.sidebar.selectbox("Select Language", ("English", "Kannada", "Telugu", "Tamil", "Hindi"))

# Sidebar with page selection
page = st.sidebar.selectbox("Select Page", ("Disease Detection", "Pathologist Info"))

# App title and description
if page == "Disease Detection":
   st.title(display_text('Rice Crop Disease Detection', language))
   st.write(display_text('Rice crop disease detection involves identifying and diagnosing various diseases that can affect rice plants, leading to reduced yield and quality. Early detection is crucial for effective management and prevention of widespread damage. Traditional methods of disease detection involve visual inspection by experts, which can be time-consuming and subjective. Advances in technology have introduced automated methods using remote sensing, machine learning, and image processing techniques. These modern approaches enable rapid, accurate, and large-scale monitoring of rice fields, helping farmers implement timely and appropriate control measures to protect their crops.', language))

   # File uploader for image
   uploaded_image = st.file_uploader(display_text("Upload an image...", language), type=["jpg", "jpeg", "png"])

   if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((400, 400))
        st.image(resized_img)

    with col2:
        if st.button(display_text('Prediction', language)):
            # This needs to be connected to your model function
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'{display_text("Prediction", language)}: {str(prediction)}')

        if st.button(display_text('Check Infected Area %', language)):
            prediction = predict_image_class(model, uploaded_image, class_indices)  # Needs definition
            percentage, mask_cleaned = calculate_infected_percentage(image)  # Needs definition
            severity_score = estimate_severity(prediction, percentage)  # Needs definition
            severity_category = get_severity_category(severity_score)  # Needs definition
            st.info(f'{display_text("Infected Area", language)}: {percentage:.2f}%')
            st.info(f'{display_text("Severity Score", language)}: {severity_score} ({severity_category})')

            # Provide recommendations based on disease and severity score
            if prediction in recommendations:  # Assuming recommendations is defined elsewhere
                recommendation = recommendations[prediction][severity_score]
                st.info(f'{display_text("Recommendation", language)}: {display_text(recommendation, language)}')
            else:
                st.warning(display_text("No recommendation available for this disease.", language))

            # Estimate and display yield growth and loss
            yield_loss_info = estimate_yield_growth_and_loss(prediction, severity_score)
            st.info(f'{display_text("Estimated Yield Growth and Loss", language)}: {yield_loss_info}')
                
    st.info(display_text("Want to check another crop? Simply upload a new image above.", language))

elif page == "Pathologist Info":
    st.title("Pathologist Information")

    # List of images to display (use your own image paths here)
    images = ["INFO.1.png", "INFO.2.png", "INFO.3.png", "INFO.4.png","INFO.5.png"]

    # Display multiple images in a grid layout
    cols = st.columns(2)  # Create two columns

    for idx, img_path in enumerate(images):
        with cols[idx % 2]:  # Alternate between the two columns
            img = Image.open(img_path)
            st.image(img, caption=f"Image {idx + 1}", use_column_width=True)
  
# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #000000;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p> Rice Crop Disease Detection &copy; 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)