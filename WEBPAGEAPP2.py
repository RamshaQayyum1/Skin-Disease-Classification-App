import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import streamlit as st
from PIL import Image

# Load the model
model = tf.keras.models.load_model("skin_disease_model_set.h5")

# Class labels
class_labels = ['Nevus', 'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 
                'melanoma', 'pigmented benign keratosis', 'squamous cell carcinoma', 'vascular lesion']


# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to 224x224
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to generate Grad-CAM
def grad_cam(model, img_array, layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Forward pass
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        # Extract the tensor from the list
        predictions = predictions[0]  # predictions is a list; get the tensor
        
        # Ensure pred_index is valid
        if pred_index is None:
           pred_index = np.argmax(predictions)
  # Get the predicted class index
        
        # Specific class output
        class_output = predictions[pred_index]  # No need for [0, pred_index]
  # Correct indexing
    
    # Backpropagation
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted feature maps
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# Function to overlay Grad-CAM on the image
def overlay_gradcam(img, heatmap, alpha=0.4):
    img = np.array(img)
    img = cv2.resize(img, (224, 224))  # Resize to match the input size of the model
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Convert heatmap to uint8
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply jet color map
    
    overlayed_image = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)  # Overlay heatmap on image
    return overlayed_image

# Streamlit interface
st.title("Skin Condition Detection with Grad-CAM")
st.markdown("Upload an image to predict the skin condition and generate a Grad-CAM heatmap.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)  # Open the uploaded image
    
    # Preprocess the image
    img_array = preprocess_image(img)
    
    # Get predictions
    predictions = model.predict(img_array)
    predictions = predictions[0]  # Extract the tensor from the list
    pred_index = np.argmax(predictions)
    predicted_class = class_labels[pred_index]
    confidence = predictions[pred_index] * 100  # Confidence percentage
    
    # Show the prediction result
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    
    # Generate Grad-CAM heatmap
    heatmap = grad_cam(model, img_array, layer_name="conv2d")  # Use your specific convolutional layer name
    
    # Overlay the heatmap on the image
    overlayed_image = overlay_gradcam(img, heatmap)
    
    # Display the heatmap on the image
    st.image(overlayed_image, caption="Grad-CAM Heatmap", use_column_width=True)

