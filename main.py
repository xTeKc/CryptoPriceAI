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

start_time = dt.datetime("INPUT START TIME") #set start time from creation date
end_time = dt.datetime("INPUT END TIME") #set end time to current date

data = web.DataReader(f"{crypto} - {fiat}", start_time, end_time)

#data prep
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

prediction_range = "IMPL PREDICTION RANGE" #set from creation date

x_train, y_train = [], []

for x in range(prediction_range, len(scaled_data)):
	x_train.append(scaled_data[x_prediction_days:x, 0])
	y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#neural network
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0, 2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0, 2))
model.add(LSTM(units=50))
model.add(Dropout(0, 2))
model.Dense(units=1)

model.compile(optimizer="", loss="")
model.fit(x_train, y_train, epochs=25, batch_size=32)


#test model
test_start = dt.datetime("INPUT START TIME")
test_end = dt.datetime.now()

test_data = web.DataReader(f"{crypto} - {fiat}", test_start, test_end)
real_prices = test_data["Close"].values

dataset_total = pd.concat((data["Close"], test_data["Close"]), axis=0)

model_inputs = dataset_total[len(dataset_total) - len(test_data) - prediction_range:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_range, len(model_inputs)):
	x_test.append(model_inputs[x_prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(real_prices, color="black", label="Real Prices")
plt.plot(prediction_prices, color="green", label="Predicted Prices")
plt.title(f"{crypto} price prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()

#predict next day?
real_data = [model_inputs[len(model_inputs) + 1 - prediction_range:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1)) 










































