import csv
import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

csv_path = r"C:\Users\mktha\Documents\projects\felix\data\RawDataMag(newdataset).csv"

df1 = pd.read_csv(csv_path, header=0, index_col=0)
# drop the first column
# df1 = df1.drop(df1.columns[0], axis=1)
print(df1.head())

# get the headers
headers = df1.columns.values.tolist()
print(headers)


param1 = df1['eps1 ']
param2 = df1['eps2 ']
param3 = df1['eps3 ']
param4 = df1['eps4 ']
param5 = df1['t1 [mm]']
param6 = df1['t2 [mm]']
param7 = df1['t3 [mm]']
param8 = df1['t4 [mm]']
# get the columns from 9 to the end
values = df1.iloc[:, 8:]

# convert to numpy array
param1 = param1.values
param2 = param2.values
param3 = param3.values
param4 = param4.values
param5 = param5.values
param6 = param6.values
param7 = param7.values
param8 = param8.values
values = values.to_numpy()
# values = values[:, 50:51]

# standardize
param1_scaler = preprocessing.StandardScaler()
param1 = param1_scaler.fit_transform(param1.reshape(-1, 1))
param2_scaler = preprocessing.StandardScaler()
param2 = param2_scaler.fit_transform(param2.reshape(-1, 1))
param3_scaler = preprocessing.StandardScaler()
param3 = param3_scaler.fit_transform(param3.reshape(-1, 1))
param4_scaler = preprocessing.StandardScaler()
param4 = param4_scaler.fit_transform(param4.reshape(-1, 1))
param5_scaler = preprocessing.StandardScaler()
param5 = param5_scaler.fit_transform(param5.reshape(-1, 1))
param6_scaler = preprocessing.StandardScaler()
param6 = param6_scaler.fit_transform(param6.reshape(-1, 1))
param7_scaler = preprocessing.StandardScaler()
param7 = param7_scaler.fit_transform(param7.reshape(-1, 1))
param8_scaler = preprocessing.StandardScaler()
param8 = param8_scaler.fit_transform(param8.reshape(-1, 1))
values_scaler = preprocessing.StandardScaler()
values = values_scaler.fit_transform(values)

inputs = np.column_stack((param1, param2, param3, param4, param5, param6, param7, param8))
# split data
X_train, X_test, y_train, y_test = train_test_split(inputs, values, test_size=0.05, random_state=42)

num_inputs = X_train.shape[1]
num_outputs = y_train.shape[1]

dropout_rate = 0.2
hidden_dim = 1024
hidden_layers = 15
input_layer = tf.keras.Input(shape=(num_inputs,))
x = layers.Dense(hidden_dim, activation='relu')(input_layer)
x = layers.LayerNormalization()(x)
x = layers.Dropout(dropout_rate)(x)
for layer in range(hidden_layers - 1):
    residual = x
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    # x = layers.Dense(hidden_dim, activation='relu')(x)
    # x = layers.LayerNormalization()(x)
    # x = layers.Dropout(dropout_rate)(x)
    x = layers.Add()([x, residual])
output_layer = layers.Dense(num_outputs)(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='../models/best_model.h5', monitor='val_loss',
                                                 save_best_only=True, verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
model.fit(X_train, y_train, epochs=300, batch_size=16, validation_data=(X_test, y_test),
          callbacks=[lr_callback, cp_callback])

# plot the loss
import matplotlib.pyplot as plt
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# predict
model = tf.keras.models.load_model('../models/best_model.h5')
y_pred = model.predict(X_test)
y_pred = values_scaler.inverse_transform(y_pred)
y_test = values_scaler.inverse_transform(y_test)


# calculate abs error along axis 1
error_abs = np.abs(y_pred - y_test)
square_error = np.square(y_pred - y_test)
mse = np.mean(np.square(error_abs))
mae = np.mean(error_abs)
print("MSE: ", mse)
print("MAE: ", mae)
