import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow_probability import distributions as tfd

csv_path = r"C:\Users\mktha\Documents\projects\felix\data\data.csv"

df = pd.read_csv(csv_path)



param1 = df['param1']
param2 = df['param2']
param3 = df['param3']
values = df['values_list']

# # convert to numpy array
param1 = param1.values
param2 = param2.values
param3 = param3.values
values = values.to_list()
values = [eval(values[i]) for i in range(len(values))]
for row in values:
    for i in range(len(row)):
        row[i] = float(row[i])
values = np.array(values)

# print stats about data
print(f'param1 mean: {param1.mean()}, std: {param1.std()}, min: {param1.min()}, max: {param1.max()}')
print(f'param2 mean: {param2.mean()}, std: {param2.std()}, min: {param2.min()}, max: {param2.max()}')
print(f'param3 mean: {param3.mean()}, std: {param3.std()}, min: {param3.min()}, max: {param3.max()}')
print(f'values mean: {values.mean()}, std: {values.std()}, min: {values.min()}, max: {values.max()}')

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
X_train, X_test, y_train, y_test = train_test_split(values, np.column_stack((param1, param2, param3)),
                                                    test_size=0.2, random_state=42)
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

def train_FCN(hidden_layers, hidden_dim, dropout_rate):
    input_layer = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation='relu')(input_layer)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    for layer in range(hidden_layers - 1):
        residual = x
        x = layers.Dense(hidden_dim, activation='relu')(x)
        x = layers.Add()([x, residual])
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
    output_layer = layers.Dense(output_dim)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='../models/best_model.h5', monitor='val_loss',
                                                     save_best_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    model.fit(X_train, y_train, epochs=200, batch_size=8, validation_data=(X_test, y_test),
              callbacks=[lr_callback, cp_callback, es_callback])

train_FCN(hidden_layers=5, hidden_dim=2048, dropout_rate=0.0)
# predict
model = tf.keras.models.load_model('../models/best_model.h5')
y_pred = model.predict(X_test)
# unstack Y_test and Y_pred
y_test_param1 = y_test[:, 0]
y_test_param2 = y_test[:, 1]
y_test_param3 = y_test[:, 2]
y_pred_param1 = y_pred[:, 0]
y_pred_param2 = y_pred[:, 1]
y_pred_param3 = y_pred[:, 2]
# unstandardize
param1_pred = param1_scaler.inverse_transform(y_pred[:, 0].reshape(-1, 1))
param2_pred = param2_scaler.inverse_transform(y_pred[:, 1].reshape(-1, 1))
param3_pred = param3_scaler.inverse_transform(y_pred[:, 2].reshape(-1, 1))
param1_test = param1_scaler.inverse_transform(y_test[:, 0].reshape(-1, 1))
param2_test = param2_scaler.inverse_transform(y_test[:, 1].reshape(-1, 1))
param3_test = param3_scaler.inverse_transform(y_test[:, 2].reshape(-1, 1))

# calculate abs error for each parameter
param1_abs_error = np.abs(param1_pred - param1_test)
param2_abs_error = np.abs(param2_pred - param2_test)
param3_abs_error = np.abs(param3_pred - param3_test)
# calculate mean abs error for each parameter
param1_mean_abs_error = np.mean(param1_abs_error)
param2_mean_abs_error = np.mean(param2_abs_error)
param3_mean_abs_error = np.mean(param3_abs_error)
# calulate mse for each parameter
param1_mse = np.mean(np.square(param1_pred - param1_test))
param2_mse = np.mean(np.square(param2_pred - param2_test))
param3_mse = np.mean(np.square(param3_pred - param3_test))
# clauculate percentage error for each parameter
param1_percentage_error = np.mean(np.abs((param1_pred - param1_test) / param1_test)) * 100
param2_percentage_error = np.mean(np.abs((param2_pred - param2_test) / param2_test)) * 100
param3_percentage_error = np.mean(np.abs((param3_pred - param3_test) / param3_test)) * 100
# print results
print('param1 mean abs error: ', param1_mean_abs_error)
print('param2 mean abs error: ', param2_mean_abs_error)
print('param3 mean abs error: ', param3_mean_abs_error)
print('param1 mse: ', param1_mse)
print('param2 mse: ', param2_mse)
print('param3 mse: ', param3_mse)
print('param1 percentage error: ', param1_percentage_error)
print('param2 percentage error: ', param2_percentage_error)
print('param3 percentage error: ', param3_percentage_error)

