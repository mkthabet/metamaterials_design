import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np


# csv_path = r"C:\Users\mktha\Documents\projects\felix\data\data.csv"
#
# df = pd.read_csv(csv_path)
#
#
#
# param1 = df['param1']
# param2 = df['param2']
# param3 = df['param3']
# values = df['values_list']
#
# # convert to numpy array
# param1 = param1.values
# param2 = param2.values
# param3 = param3.values
# values = values.to_list()
# values = [eval(values[i]) for i in range(len(values))]
# for row in values:
#     for i in range(len(row)):
#         row[i] = float(row[i])
# values = np.array(values)
#
# # standardize
# param1_scaler = preprocessing.StandardScaler()
# param1 = param1_scaler.fit_transform(param1.reshape(-1, 1))
# param2_scaler = preprocessing.StandardScaler()
# param2 = param2_scaler.fit_transform(param2.reshape(-1, 1))
# param3_scaler = preprocessing.StandardScaler()
# param3 = param3_scaler.fit_transform(param3.reshape(-1, 1))
# values_scaler = preprocessing.StandardScaler()
# values = values_scaler.fit_transform(values)
#
# # split data
# X_train, X_test, y_train, y_test = train_test_split(np.column_stack((param1, param2, param3)), values,
#                                                     test_size=0.2, random_state=42)
#
# # predict
# model = tf.keras.models.load_model('../models/best_model.h5')
# y_pred = model.predict(X_test)
# y_pred = values_scaler.inverse_transform(y_pred)
# y_test = values_scaler.inverse_transform(y_test)
# # unstack X_test
# param1_test = X_test[:, 0]
# param2_test = X_test[:, 1]
# param3_test = X_test[:, 2]
# # unstandardize
# param1_test = param1_scaler.inverse_transform(param1_test.reshape(-1, 1))
# param2_test = param2_scaler.inverse_transform(param2_test.reshape(-1, 1))
# param3_test = param3_scaler.inverse_transform(param3_test.reshape(-1, 1))
#
# # calculate abs error along axis 1
# error_abs = np.abs(y_pred - y_test)
# square_error = np.square(y_pred - y_test)
# mse = np.mean(np.square(error_abs))
# mae = np.mean(error_abs)
# print("MSE: ", mse)
# print("MAE: ", mae)
#
# # create a dataframe with the following column names: param1, param2, param3, values_actual, values_predicted, abs_error, squared_error
# df = pd.DataFrame({'param1': param1_test[:, 0], 'param2': param2_test[:, 0], 'param3': param3_test[:, 0],
#                      'values_actual': y_test.tolist(), 'values_predicted': y_pred.tolist(),
#                    'abs_error': error_abs.tolist(), 'squared_error': square_error.tolist()})

# df.to_csv("../data/results.csv", index=False)
df = pd.read_csv("../data/results.csv")


for index, row in df.iterrows():
    actual = row['values_actual']
    # get numpy array from string
    actual = actual[1:-1]
    actual = actual.split()
    actual = [float(i.replace(',', '')) for i in actual]
    predicted = row['values_predicted']
    predicted = predicted[1:-1]
    predicted = predicted.split()
    predicted = [float(i.replace(',', '')) for i in predicted]
    plt.plot(actual, label='actual')
    plt.plot(predicted, label='predicted')
    plt.title(f'param1: {row["param1"]}, param2: {row["param2"]}, param3: {row["param3"]}')
    plt.legend()
    plt.show()

