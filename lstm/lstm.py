import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pylt

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
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+5]]
        X.append(row)
        label = df_as_np[i+5]
        y.append(label)
    
    return np.array(X), np.array(y)

WINDOW_SIZE = 5
X, y = df_to_X_y(temp, WINDOW_SIZE)
print(X.shape, y.shape)