import streamlit as st
from preprocessing import preprocess_input
from model_loader import load_model

st.title("🧠 Dự đoán Tổng Tiêu thụ Điện (Regression)")

# Nhập các thông số tương ứng
temp = st.number_input("Nhiệt độ (°C)", value=25.0)
humidity = st.number_input("Độ ẩm (%)", value=60.0)
pressure = st.number_input("Áp suất (hPa)", value=1013.0)
wind_speed = st.number_input("Tốc độ gió (m/s)", value=3.0)
rain_1h = st.number_input("Lượng mưa 1h", value=0.0)
snow_3h = st.number_input("Lượng tuyết 3h", value=0.0)
clouds_all = st.number_input("Cloud cover (%)", value=50.0)
hour = st.number_input("Giờ trong ngày (0–23)", value=12, min_value=0, max_value=23)
day_of_week = st.number_input("Thứ (0=Thứ 2, …6=Chủ nhật)", value=2, min_value=0, max_value=6)
month = st.number_input("Tháng (1–12)", value=6, min_value=1, max_value=12)

if st.button("Dự đoán"):
    X_input = preprocess_input(temp, humidity, pressure, wind_speed,
                               rain_1h, snow_3h, clouds_all, hour, day_of_week, month)
    model = load_model()
    pred = model.predict(X_input)
    st.success(f"Dự đoán: **{pred[0]:,.2f} MW**")
