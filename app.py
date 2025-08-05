import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Phân tích tác động của thời tiết đến tiêu thụ điện năng")

# Load dữ liệu
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/phantuan1311/KhaiPhaDL/main/filtered_data.csv"
    df = pd.read_csv(url)
    df['time'] = pd.to_datetime(df['time'])
    df['temp_c'] = df['temp'] - 273.15
    return df

df = load_data()

# Hiển thị bảng dữ liệu
st.subheader("Dữ liệu đã xử lý")
st.dataframe(df.head())

# Biểu đồ phân tán giữa từng biến thời tiết và mức tiêu thụ
st.subheader("Mối quan hệ giữa các yếu tố thời tiết và tiêu thụ điện")

weather_features = ['temp_c', 'humidity', 'pressure', 'wind_speed', 'rain_1h', 'snow_3h', 'clouds_all']

for feature in weather_features:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=feature, y='total load actual', alpha=0.3, ax=ax)
    ax.set_title(f"{feature} vs Total Load Actual")
    ax.set_xlabel(feature)
    ax.set_ylabel("Tiêu thụ điện năng (MW)")
    st.pyplot(fig)

# Biểu đồ tương quan
st.subheader("Ma trận tương quan giữa các biến")
fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
corr = df[weather_features + ['total load actual']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
st.pyplot(fig_corr)
