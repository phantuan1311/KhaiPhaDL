import streamlit as st
from model_loader import load_model
from preprocessing import preprocess_input

st.title("Dự đoán điện năng tiêu thụ")

# --- Chọn mô hình ---
model_option = st.selectbox("Chọn mô hình dự đoán", ["Random Forest", "XGBoost"])

# --- Nhập thông số thời tiết ---
temp = st.number_input("Nhiệt độ (°C)", value=25.0)
humidity = st.number_input("Độ ẩm (%)", value=60.0)
wind_speed = st.number_input("Tốc độ gió (m/s)", value=3.0)

if st.button("Dự đoán"):
    model_path = "random_forest.pkl" if model_option == "Random Forest" else "xgboost_model.pkl"
    model = load_model(model_path)

    input_data = preprocess_input(temp, humidity, wind_speed)
    prediction = model.predict(input_data)

    st.success(f"Mô hình {model_option} dự đoán: {prediction[0]:,.2f} MW")
