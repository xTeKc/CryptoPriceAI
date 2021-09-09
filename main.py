import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt 

from sklearn.preprocessing import MinMaxScaler
from tenserflow.keras.layers import Dense, Dropout, LSTM
from tenserflow.keras.models import Sequential

crypto = "BTC"
fiat = "USD"

start_time = dt.datetime("INPUT START TIME")
end_time = dt.datetime("INPUT END TIME")

data = web.DataReader(f"{crypto} - {fiat}", start_time, end_time)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

prediction_range = "IMPL PREDICTION RANGE"