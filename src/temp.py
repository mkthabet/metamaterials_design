import csv
import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

csv_path = r"C:\Users\mktha\Documents\projects\felix\data\data.csv"

df = pd.read_csv(csv_path)



param1 = df['param1']
param2 = df['param2']
param3 = df['param3']
values = df['values_list']

# convert to numpy array
param1 = param1.values
param2 = param2.values
param3 = param3.values
values = values.to_list()
values = [eval(values[i]) for i in range(len(values))]
for row in values:
    for i in range(len(row)):
        row[i] = float(row[i])
values = np.array(values)

# standardize
param1_scaler = preprocessing.StandardScaler()
param1 = param1_scaler.fit_transform(param1.reshape(-1, 1))
param2_scaler = preprocessing.StandardScaler()
param2 = param2_scaler.fit_transform(param2.reshape(-1, 1))
param3_scaler = preprocessing.StandardScaler()
param3 = param3_scaler.fit_transform(param3.reshape(-1, 1))
values_scaler = preprocessing.StandardScaler()
values = values_scaler.fit_transform(values)

# split data
X_train, X_test, y_train, y_test = train_test_split(np.column_stack((param1, param2, param3)), values, test_size=0.2, random_state=42)


dropout_rate = 0.0
inputs = np.column_stack((param1, param2, param3))
input_layer = tf.keras.Input(shape=(3,))
x = layers.Dense(64, activation='relu')(input_layer)
x = layers.Dropout(dropout_rate)(x)
# x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(dropout_rate)(x)
# x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(dropout_rate)(x)
# x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(dropout_rate)(x)
# x = layers.BatchNormalization()(x)
x = layers.Dense(1001, activation='relu')(x)
x = layers.Reshape((1001, 1))(x)
x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
# x = layers.BatchNormalization()(x)
output_layer = layers.Dense(1)(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_callback])

# predict
y_pred = model.predict(X_test)
y_pred = values_scaler.inverse_transform(y_pred)
y_test = values_scaler.inverse_transform(y_test)
y_pred = y_pred.reshape(-1)
# calculate error
error_abs = np.abs(y_pred - y_test)
mse = np.mean(np.square(error_abs))
mae = np.mean(error_abs)
print("MSE: ", mse)
print("MAE: ", mae)