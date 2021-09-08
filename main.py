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

