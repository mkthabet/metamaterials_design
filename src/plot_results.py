import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from lib.schedulers import WarmUpCosine

csv_path = r"data/Tyx_mag(Final).csv"
df_mag = pd.read_csv(csv_path)

params = df_mag.iloc[:, 0:8].values
# get the columns from 9 to the end
values = df_mag.iloc[:, 8:].values
# values = values[:, :100]
# standardize
params_scaler = preprocessing.StandardScaler()
params = params_scaler.fit_transform(params)
values_scaler = preprocessing.StandardScaler()
values = values_scaler.fit_transform(values)

# split data
X_train, X_test, y_train, y_test = train_test_split(params, values, test_size=0.1, random_state=42)

# predict
model = tf.keras.models.load_model('models/cp_hearty-sweep-29.h5', custom_objects={'WarmUpCosine': WarmUpCosine})
y_pred = model.predict(X_test)
y_pred = values_scaler.inverse_transform(y_pred)
y_test = values_scaler.inverse_transform(y_test)
# unstandardize
x_test = params_scaler.inverse_transform(X_test)

# calculate abs error along axis 1
error_abs = np.abs(y_pred - y_test)
square_error = np.square(y_pred - y_test)
mse = np.mean(np.square(error_abs))
mae = np.mean(error_abs)
print("MSE: ", mse)
print("MAE: ", mae)

# create a dataframe with the following column names: param1, param2, param3, values_actual, values_predicted, abs_error, squared_error
df = pd.DataFrame({'eps1': x_test[:, 0], 'eps2': x_test[:, 1], 'eps3': x_test[:, 2], 'eps4': x_test[:, 3], 't1': x_test[:, 4], 't2': x_test[:, 5], 't3': x_test[:, 6], 't4': x_test[:, 7],
                     'values_actual': y_test.tolist(), 'values_predicted': y_pred.tolist(),
                   'abs_error': error_abs.tolist(), 'squared_error': square_error.tolist()})

df.to_csv("data/results.csv", index=False)
df = pd.read_csv("data/results.csv")


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
    plt.title(f'eps1: {row["eps1"]}, eps2: {row["eps2"]}, eps3: {row["eps3"]}, eps4: {row["eps4"]}, t1: {row["t1"]}, t2: {row["t2"]}, t3: {row["t3"]}, t4: {row["t4"]}')
    plt.legend()
    plt.show()

