import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# ----------------------
# Page configuration
# ----------------------
st.set_page_config(
    page_title="AirDraw – Digit Recognition ✍️",
    layout="centered",
    page_icon="🎨"
)

# Title with emoji and color
st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>🎨 AirDraw – Digit Recognition ✍️</h1>
    <p style='text-align: center; font-size:16px; color:#555;'>Upload your IMU CSV data to see which digit you drew!</p>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Load model and scaler
# ----------------------
@st.cache_resource
def load_artifacts():
    model = load_model(r"D:\project\FINALPROJECT\models\airdraw_cnn_lstm.h5")
    scaler = joblib.load(r"D:\project\FINALPROJECT\processed\scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ----------------------
# File uploader in a colorful card
# ----------------------
st.markdown(
    """
    <div style='background: linear-gradient(90deg, #FFD6D6, #FFEDAA); 
                border-radius:15px; padding:25px; text-align:center; box-shadow: 2px 2px 10px rgba(0,0,0,0.15);'>
        <h3>📂 Upload Your CSV</h3>
        <p style='font-size:14px; color:#555;'>Make sure it has columns: ax, ay, az, gx, gy, gz and at least 200 rows</p>
    </div>
    """,
    unsafe_allow_html=True
)

file = st.file_uploader(
    "⬇️ Drag & drop your IMU CSV here",
    type="csv",
    label_visibility="collapsed"
)

# ----------------------
# Handle uploaded file
# ----------------------
if file is not None:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
    else:
        required_cols = ["ax", "ay", "az", "gx", "gy", "gz"]

        if not all(col in df.columns for col in required_cols):
            st.error("❌ CSV must contain columns: ax, ay, az, gx, gy, gz")
        elif len(df) < 200:
            st.error("❌ CSV must contain at least 200 rows")
        else:
            # Preprocess input
            X = df[required_cols].values[:200]
            X = scaler.transform(X).reshape(1, 200, 6)

            # Prediction
            pred = model.predict(X, verbose=0)
            digit = np.argmax(pred)
            confidence = pred[0][digit] * 100

            # Colorful card for prediction
            st.markdown(
                f"""
                <div style='background: linear-gradient(90deg, #D6FFB3, #80FFEA); 
                            padding:25px; border-radius:20px; text-align:center; margin-top:20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);'>
                    <h2>✨ Predicted Digit: <span style='color:#FF5733; font-size:48px;'>{digit}️⃣</span></h2>
                    <p style='font-size:20px; color:#333;'>Confidence: <strong>{confidence:.2f}%</strong> 💯</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Probability distribution for all digits
            st.subheader("📊 Confidence for all digits")
            conf_df = pd.DataFrame({
                "Digit": [f"{i}️⃣" for i in range(10)],
                "Confidence (%)": (pred[0]*100).round(2)
            })

            st.bar_chart(conf_df.set_index("Digit"), use_container_width=True)
