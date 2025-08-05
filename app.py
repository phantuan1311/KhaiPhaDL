import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình giao diện
st.set_page_config(page_title="Phân tích tác động thời tiết tới tiêu thụ điện", layout="wide")

# Tiêu đề chính
st.title("⚡ Phân tích ảnh hưởng của điều kiện thời tiết tới mức tiêu thụ điện")

st.markdown("""
Ứng dụng này giúp trực quan hóa mối quan hệ giữa các yếu tố thời tiết (nhiệt độ, độ ẩm, tốc độ gió, v.v...) với lượng tiêu thụ điện năng thực tế. Dữ liệu được lấy từ tập hợp thời gian thực của châu Âu, đã qua xử lý.
""")

# Tải dữ liệu đã xử lý từ GitHub hoặc local (tùy môi trường deploy)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/phantuan1311/KhaiPhaDL/main/merged_data.csv"
    df = pd.read_csv(url)
    df['time'] = pd.to_datetime(df['time'])
    return df

df = load_data()

# --- Giao diện chọn lọc ---
st.sidebar.header("⚙️ Tuỳ chọn hiển thị")
selected_feature = st.sidebar.selectbox("Chọn biến thời tiết để phân tích", 
    ['temp_c', 'humidity', 'pressure', 'wind_speed', 'rain_1h', 'snow_3h', 'clouds_all'])

# --- Phân tích hồi quy tuyến tính ---
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df[[selected_feature]]
y = df['total load actual']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# --- Hiển thị biểu đồ ---
st.subheader(f"📊 Mối quan hệ giữa `{selected_feature}` và tiêu thụ điện")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=selected_feature, y='total load actual', data=df, alpha=0.4, ax=ax)
sns.lineplot(x=df[selected_feature], y=y_pred, color='red', label='Hồi quy tuyến tính', ax=ax)
ax.set_xlabel(selected_feature)
ax.set_ylabel("Tiêu thụ điện (MW)")
ax.set_title(f"R² = {r2:.4f}")
ax.grid(True)
st.pyplot(fig)

# --- Hiển thị hệ số tương quan ---
st.subheader("📈 Phân tích tương quan Pearson giữa các biến")
correlation = df[['total load actual', 'temp_c', 'humidity', 'pressure', 'wind_speed', 
                  'rain_1h', 'snow_3h', 'clouds_all']].corr()
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
st.pyplot(fig2)

# --- Giải thích ---
st.markdown("#### 📌 Giải thích:")
st.markdown("""
- **R² (R-squared)** thể hiện mức độ giải thích của biến thời tiết tới biến tiêu thụ điện năng.
- **Biểu đồ phân tán + đường hồi quy** giúp bạn nhìn thấy xu hướng.
- **Ma trận tương quan** giúp đánh giá ảnh hưởng tổng thể giữa nhiều yếu tố thời tiết với lượng điện tiêu thụ.
""")
