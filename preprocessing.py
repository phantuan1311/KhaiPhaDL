import numpy as np

def preprocess_input(temp, humidity, wind_speed):
    # Định dạng input cho đúng với model
    return np.array([[temp, humidity, wind_speed]])
