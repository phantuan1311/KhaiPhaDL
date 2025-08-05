import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input(temp, humidity, pressure, wind_speed, rain_1h,
                     snow_3h, clouds_all, hour, day_of_week, month):
    scal = pd.read_pickle("scaler.pkl")
    arr = np.array([[temp, humidity, pressure, wind_speed,
                     rain_1h, snow_3h, clouds_all,
                     hour, day_of_week, month]])
    return scal.transform(arr)
