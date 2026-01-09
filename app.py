import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Title
st.title("🌾 Crop Recommendation System")
st.markdown("Predict the **best crop** based on soil and climate conditions")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("crop recommendation.csv")
    return df

df = load_data()

# Show dataset
if st.checkbox("📊 Show Dataset"):
    st.dataframe(df)

# Encode target column
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Features & Target
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: **{accuracy * 100:.2f}%**")

# Sidebar inputs
st.sidebar.header("🧪 Enter Soil & Climate Values")

N = st.sidebar.number_input("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.number_input("Phosphorus (P)", 0, 140, 50)
K = st.sidebar.number_input("Potassium (K)", 0, 200, 50)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.sidebar.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 400.0, 100.0)

# Prediction
if st.button("🌱 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop_name = le.inverse_transform(prediction)

    st.balloons()
    st.subheader("🌾 Recommended Crop:")
    st.success(f"**{crop_name[0]}**")
