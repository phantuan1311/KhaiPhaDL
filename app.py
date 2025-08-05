import streamlit as st
import numpy as np
import pickle

# Load mô hình đã huấn luyện
with open("random_forest.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Ảnh hưởng của thời tiết đến mức tiêu thụ điện")

st.write("Thay đổi các yếu tố thời tiết để xem ảnh hưởng đến mức tiêu thụ điện:")

# Các thanh trượt
temp_c = st.slider("Nhiệt độ (°C)", -10.0, 40.0, 20.0)
humidity = st.slider("Độ ẩm (%)", 0, 100, 50)
pressure = st.slider("Áp suất (hPa)", 900, 1050, 1013)
wind_speed = st.slider("Tốc độ gió (m/s)", 0.0, 25.0, 3.0)
rain_1h = st.slider("Lượng mưa 1h (mm)", 0.0, 50.0, 0.0)
snow_3h = st.slider("Tuyết rơi 3h (mm)", 0.0, 50.0, 0.0)
clouds_all = st.slider("Mây che phủ (%)", 0, 100, 20)
hour = st.slider("Giờ trong ngày", 0, 23, 12)
day_of_week = st.slider("Thứ trong tuần (0=Thứ 2)", 0, 6, 2)
month = st.slider("Tháng", 1, 12, 8)

# Biến môi trường năng lượng nếu có
price_actual = st.slider("Giá điện (€/MWh)", 0.0, 300.0, 50.0)
gen_solar = st.slider("Điện mặt trời (MW)", 0.0, 5000.0, 500.0)
gen_wind = st.slider("Điện gió (MW)", 0.0, 10000.0, 2000.0)
gen_coal = st.slider("Điện than (MW)", 0.0, 20000.0, 5000.0)
gen_hydro = st.slider("Thủy điện bơm tích trữ (MW)", 0.0, 5000.0, 1000.0)

# Tạo mảng đầu vào
input_data = np.array([[temp_c, humidity, pressure, wind_speed, rain_1h, snow_3h, clouds_all,
                        hour, day_of_week, month, price_actual, gen_solar, gen_wind, gen_coal, gen_hydro]])

# Dự đoán
prediction = model.predict(input_data)[0]

st.subheader("👉 Mức tiêu thụ điện dự đoán:")
st.metric(label="Total Load Actual (MW)", value=f"{prediction:,.2f}")
