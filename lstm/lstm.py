import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pylt
from tensorflow import keras

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
df = df[5::6]
df.head()
df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df[:26]

temp = df['T (degC)']


# pylt.plot(temp)
# pylt.show()


def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + 5]]
        X.append(row)
        label = df_as_np[i + 5]
        y.append(label)

    return np.array(X), np.array(y)


WINDOW_SIZE = 5

X, y = df_to_X_y(temp, WINDOW_SIZE)
# print(X.shape, y.shape)

X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]

# print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)


from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((5, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

# model1.summary()
#
# cp = ModelCheckpoint('model1/', save_best_only=True)
# model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
#
# model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

from keras.models import load_model
model1 = load_model('model1/')

# For training data
# train_predictions = model1.predict(X_train).flatten()
# train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})

# For validation data
# train_predictions = model1.predict(X_val).flatten()
# train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_val})

# For test data
train_predictions = model1.predict(X_test).flatten()
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_test})

pylt.plot(train_results['Train Predictions'][:100])
pylt.plot(train_results['Actuals'][:100])
pylt.show()
