import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import gdown  # optional, if you download from Google Drive

# ------------------------------------------------------------
# 0. Download the model if not present (optional)
# ------------------------------------------------------------
model_path = 'mobileNetV2_FMD.h5'
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        # Replace with your actual model URL or Google Drive file ID
        # Example for Google Drive (you need gdown installed)
        # url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
        # gdown.download(url, model_path, quiet=False)
        #
        # If using a direct download link, you can use requests:
        # import requests
        # r = requests.get("https://example.com/model.h5", stream=True)
        # with open(model_path, 'wb') as f:
        #     for chunk in r.iter_content(chunk_size=8192):
        #         f.write(chunk)
        st.warning("Please place the model file in the same directory or update the download logic.")
        st.stop()

# ------------------------------------------------------------
# 1. Load the model
# ------------------------------------------------------------
@st.cache_resource
def load_fmd_model():
    model = load_model(model_path)
    return model

model = load_fmd_model()
class_names = ['healthy', 'FMD']

# ------------------------------------------------------------
# 2. Preprocessing function
# ------------------------------------------------------------
def preprocess_image(img):
    """Resize and normalize image to 224x224, float32 [0,1]"""
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------------------------------------------
# 3. Grad‑CAM function (works with Keras model)
# ------------------------------------------------------------
def get_gradcam_heatmap(model, img_array, last_conv_layer_name='Conv_1'):
    """
    Generate Grad-CAM heatmap.
    Assumes model is the Sequential model with MobileNetV2 as first layer.
    """
    # Extract the base model (MobileNetV2) and the top layers
    base_model = model.layers[0]
    # Get the convolutional layer from the base model
    conv_layer = base_model.get_layer(last_conv_layer_name)

    # Create a model that outputs both the conv layer output and the final prediction
    grad_model = tf.keras.Model(
        [base_model.input],
        [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_array)
        loss = predictions[0][0]  # sigmoid output

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()

# ------------------------------------------------------------
# 4. Overlay heatmap on image
# ------------------------------------------------------------
def overlay_heatmap(img, heatmap):
    """Overlay heatmap on original image and return the resulting image (PIL)."""
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(overlay)

# ------------------------------------------------------------
# 5. Advice function
# ------------------------------------------------------------
def get_advice(pred_class, confidence):
    if pred_class == 1:
        return (
            "**⚠️ FMD DETECTED**

"
            "1. Isolate the affected animal immediately.
"
            "2. Disinfect your hands, clothing, and equipment.
"
            "3. Do not move any cattle on or off the farm.
"
            "4. Provide soft feed and clean water.
"
            "5. Contact a veterinarian or extension officer immediately."
        )
    else:
        return (
            "**✅ HEALTHY**

"
            "No signs of Foot and Mouth Disease detected.
"
            "Continue routine monitoring and maintain good biosecurity practices.
"
            "If you notice any unusual signs later, consult a veterinarian."
        )

# ------------------------------------------------------------
# 6. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="FMD Detector", layout="wide")
st.title("🐄 Foot and Mouth Disease Detector")
st.markdown("Upload a photo of a cow to get an instant diagnosis with explainable AI.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    use_camera = st.checkbox("Use camera instead")
    if use_camera:
        img_file = st.camera_input("Take a picture")
    else:
        img_file = uploaded_file

    if img_file is not None:
        img = Image.open(img_file).convert('RGB')
        st.image(img, caption="Original Image", use_container_width=True)

with col2:
    st.subheader("Diagnosis & Explanation")
    if img_file is not None:
        with st.spinner("Analyzing..."):
            # Preprocess
            img_array = preprocess_image(img)

            # Predict
            prob = model.predict(img_array)[0][0]
            pred_class = 1 if prob > 0.5 else 0
            confidence = prob if pred_class == 1 else 1 - prob

            # Display result
            result_text = f"**Diagnosis:** {class_names[pred_class].upper()} "
            result_text += f"(confidence: {confidence:.1%})"
            st.write(result_text)

            # Advice
            st.markdown(get_advice(pred_class, confidence))

            # Grad-CAM heatmap
            try:
                heatmap = get_gradcam_heatmap(model, img_array)
                overlay_img = overlay_heatmap(img, heatmap)
                st.image(overlay_img, caption="Grad‑CAM Overlay", use_container_width=True)
            except Exception as e:
                st.warning(f"Heatmap could not be generated: {e}")

            # Contact button
            st.markdown("---")
            st.subheader("Need help?")
            if st.button("📞 Contact Veterinary Officer"):
                # For demonstration, we show a number. In practice, you could open a link.
                st.info("Call the nearest veterinary office: +268 1234 5678")
                # Alternatively, you could provide a clickable link:
                # st.markdown("[Call Veterinary Officer](tel:+26812345678)")

# ------------------------------------------------------------
# 7. Sidebar (info)
# ------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "This tool uses a deep learning model (MobileNetV2) trained on images "
        "from Eswatini to detect Foot and Mouth Disease in cattle. "
        "The Grad‑CAM heatmap highlights the areas the model focused on for the diagnosis."
    )
    st.markdown("**Disclaimer:** This tool is for early detection support only. "
                "Always consult a veterinarian for confirmation and treatment.")
