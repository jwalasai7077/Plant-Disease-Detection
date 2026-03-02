import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import pandas as pd
import tempfile
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🌿 Plant Disease Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-Based Leaf Disease Classification</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

model = load_model()

with open("class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = 224

# -------------------------------------------------
# SESSION STATE FOR HISTORY
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    with st.spinner("🔍 Analyzing Image... Please wait..."):
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="📷 Uploaded Image", width="stretch")

    with col2:
        st.success(f"🌱 Predicted Disease: {predicted_class}")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")

    # -------------------------------------------------
    # TOP 5 CONFIDENCE GRAPH
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Top 5 Prediction Probabilities")

    probabilities = prediction[0]
    top5_idx = np.argsort(probabilities)[-5:][::-1]

    top5_classes = [class_names[i] for i in top5_idx]
    top5_probs = probabilities[top5_idx] * 100

    df = pd.DataFrame({
        "Disease": top5_classes,
        "Confidence (%)": top5_probs
    })

    st.bar_chart(df.set_index("Disease"))

    # -------------------------------------------------
    # SAVE HISTORY
    # -------------------------------------------------
    st.session_state.history.append({
        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Disease": predicted_class,
        "Confidence (%)": round(confidence, 2)
    })

    # -------------------------------------------------
    # PDF REPORT
    # -------------------------------------------------
    def generate_pdf():
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(temp_file.name)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Plant Disease Detection Report", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Disease: {predicted_class}", styles["Normal"]))
        elements.append(Paragraph(f"Confidence: {confidence:.2f}%", styles["Normal"]))
        elements.append(Paragraph(f"Generated on: {datetime.datetime.now()}", styles["Normal"]))

        doc.build(elements)
        return temp_file.name

    if st.button("📄 Download PDF Report"):
        pdf_path = generate_pdf()
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Click to Download Report",
                data=f,
                file_name="Plant_Disease_Report.pdf",
                mime="application/pdf"
            )

# -------------------------------------------------
# PREDICTION HISTORY
# -------------------------------------------------
st.markdown("---")
st.subheader("🧪 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, width="stretch")
else:
    st.write("No predictions yet.")