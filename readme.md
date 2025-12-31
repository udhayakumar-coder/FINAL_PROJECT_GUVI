# AirDraw – Digit Recognition using IMU Data

## 📌 Project Overview
AirDraw is a machine learning project that recognizes handwritten digits drawn in air using IMU sensor data.  
The model uses accelerometer and gyroscope readings to predict digits from 0 to 9.

---

## 📂 Folder Structure
FINALPROJECT/
│
├── source/ # Raw IMU CSV files (digit-wise folders)
├── processed/ # Processed data and scaler
├── models/ # Trained CNN-LSTM model
├── data.ipynb # Data preprocessing notebook
├── app.py # Streamlit application
└── README.md
---

## 🧠 Features Used
- ax, ay, az → Accelerometer values
- gx, gy, gz → Gyroscope values

Each sample is resampled to **200 time steps**.

---

## ⚙️ Data Processing Steps
1. Read IMU CSV files
2. Select required sensor columns
3. Resample data to fixed length
4. Standardize features using StandardScaler
5. Split data into train and test sets
6. Save processed data and scaler

---

## 🤖 Model
- CNN + LSTM deep learning model
- Input shape: (200, 6)
- Output: Digit class (0–9)

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install numpy pandas scikit-learn scipy tensorflow streamlit joblib

### 2️⃣ Run preprocessing
python data.ipynb

### 3️⃣ Start Streamlit app
streamlit run app.py


📊 Prediction

Upload a CSV file containing IMU data.
The app will display the predicted digit.

📝 Notes

Ensure uploaded CSV has at least 200 rows

Column names must match: ax, ay, az, gx, gy, gz

Scaler must be the same one used during training

👤 Author

Udhayakumar
DATA SCIENCE
