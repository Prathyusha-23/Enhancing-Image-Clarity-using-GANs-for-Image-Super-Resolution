import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import pyttsx3
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ================== CONSTANTS ==================
IMG_SIZE = 128

class_names = ['akiec','bcc','bkl','df','nv','vasc','mel']

class_full_info = {
    "akiec": ("Actinic Keratoses", "Precancerous"),
    "bcc": ("Basal Cell Carcinoma", "Cancerous"),
    "bkl": ("Benign Keratosis", "Benign"),
    "df": ("Dermatofibroma", "Benign"),
    "nv": ("Melanocytic Nevus", "Benign"),
    "vasc": ("Vascular Lesion", "Benign"),
    "mel": ("Melanoma", "Cancerous")
}

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Skin Disease Detection + Enhancing Image Clarity with GANs: A Deep Learning Approach to Super-Resolution",
    page_icon="🧬",
    layout="wide"
)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model("classifier.keras")

classifier = load_classifier()

# ================== IMAGE ENHANCEMENT ==================
def enhance_clarity(img):
    """Medical style clarity enhancement keeping colors unchanged."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)

    ycrcb = cv2.merge((y, cr, cb))
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    img = cv2.bilateralFilter(img, 9, 75, 75)

    blur = cv2.GaussianBlur(img, (0,0), 3)
    sharpen = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

    return sharpen

# ================== UI ==================
st.title("🧬 Skin Disease Detection + Clarity Enhancement")
st.write("Upload a skin lesion image. The system enhances clarity and compares cancer risk before and after enhancement.")

uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    # ================== ORIGINAL PREDICTION ==================
    orig_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    orig_pred = classifier.predict(
        np.expand_dims(orig_resized.astype(np.float32)/255.0, axis=0), verbose=0
    )[0]
    orig_probs = orig_pred * 100
    orig_mel_prob = orig_probs[class_names.index("mel")]

    # ================== IMAGE ENHANCEMENT ==================
    enhanced = enhance_clarity(img_np)
    enhanced_resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    pred = classifier.predict(
        np.expand_dims(enhanced_resized.astype(np.float32)/255.0, axis=0), verbose=0
    )[0]
    enh_probs = pred * 100
    mel_prob = enh_probs[class_names.index("mel")]

    idx = np.argmax(pred)
    cls = class_names[idx]
    name, status = class_full_info[cls]

    # ================== DISPLAY IMAGES ==================
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_np, use_column_width=True)
    with col2:
        st.subheader("Enhanced Image")
        st.image(enhanced, use_column_width=True)

    # ================== RESULT + PRECAUTIONS ==================
    st.subheader("🧠 Prediction Result")
    if mel_prob > 70:
        st.error("High Risk ⚠️")
        conclusion = "Skin cancer is likely present. Please consult a dermatologist immediately."
        precautions_list = [
            "Visit a dermatologist for clinical diagnosis.",
            "Avoid direct sun exposure.",
            "Do not scratch or irritate the lesion.",
            "Monitor the size, shape, and color changes.",
            "Consider dermoscopy or biopsy if advised by a doctor."
        ]
    elif mel_prob > 30:
        st.warning("Moderate Risk 🩺")
        conclusion = "There is a possibility of skin cancer. Medical consultation is recommended."
        precautions_list = [
            "Monitor the lesion regularly.",
            "Avoid excessive sun exposure.",
            "Use sunscreen when outdoors.",
            "Seek dermatology consultation if the lesion changes."
        ]
    else:
        st.success("Low Risk ✅")
        conclusion = "Skin cancer is not present according to the model prediction."
        precautions_list = [
            "Maintain proper skin hygiene.",
            "Avoid prolonged sun exposure.",
            "Use sunscreen regularly.",
            "Monitor skin for any unusual changes.",
            "Maintain a healthy lifestyle."
        ]

    st.markdown(f"**Detected Disease:** {name}")
    st.markdown(f"**Status:** {status}")
    st.markdown(f"**Confidence:** {pred[idx]*100:.2f} %")
    st.markdown("### ✅ Final Conclusion" if mel_prob <=30 else "### ⚠️ Final Conclusion")
    st.markdown(conclusion)
    st.markdown("### Recommended Precautions")
    for p in precautions_list:
        st.write(f"- {p}")

    # ================== VOICE ==================
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)

        voice_message = f"Prediction result. Detected disease is {name}. "
        voice_message += f"Risk level is {'high' if mel_prob>70 else 'moderate' if mel_prob>30 else 'low'}. "
        voice_message += f"{conclusion} Recommended precautions are: "
        voice_message += " ".join(precautions_list)

        engine.say(voice_message)
        engine.runAndWait()
    except:
        pass

    # ================== PROBABILITY TABLE ==================
    st.subheader("Disease Probability Table")
    prob_df = pd.DataFrame({
        "Disease": [class_full_info[c][0] for c in class_names],
        "Original (%)": orig_probs,
        "Enhanced (%)": enh_probs
    })
    st.dataframe(prob_df.style.format({"Original (%)":"{:.2f}","Enhanced (%)":"{:.2f}"}))

    # ================== DIFFERENCE TABLE ==================
    st.subheader("Prediction Difference Table")
    diff_df = pd.DataFrame({
        "Disease": [class_full_info[c][0] for c in class_names],
        "Original (%)": orig_probs,
        "Enhanced (%)": enh_probs,
        "Difference (%)": enh_probs - orig_probs
    })
    st.dataframe(diff_df.style.format({"Original (%)":"{:.2f}","Enhanced (%)":"{:.2f}","Difference (%)":"{:.2f}"}))

    # ================== BAR CHART ==================
    st.subheader("Cancer Risk Comparison")
    risk_df = pd.DataFrame({
        "Image Type": ["Original","Enhanced"],
        "Melanoma Risk": [orig_mel_prob, mel_prob]
    })
    st.bar_chart(risk_df.set_index("Image Type"))

    st.subheader("Cancer Risk Difference")
    risk_change = mel_prob - orig_mel_prob
    st.write(f"Risk {'increased' if risk_change>0 else 'decreased'} by {abs(risk_change):.2f}% after enhancement")

    # ================== SCATTER PLOT ==================
    st.subheader("Scatter Plot: Risk Before vs After Enhancement")
    fig1, ax1 = plt.subplots()
    ax1.scatter(orig_probs, enh_probs, color="blue")
    for i, txt in enumerate(class_names):
        ax1.annotate(txt, (orig_probs[i], enh_probs[i]))
    ax1.set_xlabel("Original Probability (%)")
    ax1.set_ylabel("Enhanced Probability (%)")
    ax1.set_title("Disease Probability Comparison")
    st.pyplot(fig1)

    # ================== LINE PLOT ==================
    st.subheader("Line Plot: Disease Probability Change")
    fig2, ax2 = plt.subplots()
    ax2.plot(class_names, orig_probs, label="Original", marker="o")
    ax2.plot(class_names, enh_probs, label="Enhanced", marker="o")
    ax2.set_xlabel("Disease Type")
    ax2.set_ylabel("Probability (%)")
    ax2.set_title("Probability Change After Enhancement")
    ax2.legend()
    st.pyplot(fig2)

    # ================== IMAGE QUALITY METRICS ==================
    st.subheader("Image Quality Metrics")
    psnr = peak_signal_noise_ratio(orig_resized, enhanced_resized)
    ssim = structural_similarity(orig_resized, enhanced_resized, channel_axis=2)
    quality_df = pd.DataFrame({"Metric":["PSNR","SSIM"],"Value":[psnr, ssim]})
    st.table(quality_df)

    st.markdown("---")
    st.info("⚠️ Educational use only. Consult a dermatologist for medical diagnosis.")