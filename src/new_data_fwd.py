import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import wandb
import scipy.io
from lib.schedulers import WarmUpCosine

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
    {
        'batch_size': {'values': [256, 512, 1024]},
        'lr_start': {'max': 0.0001, 'min': 0.00001},
        'lr_max': {'max': 0.01, 'min': 0.005},
        'warmup_steps': {'values': [1000, 2000, 3000]},
        'hidden_dim': {'values': [256, 512, 1024]},
        # 'dropout': {'values': [0.0, 0.1, 0.2, 0.3]},
        'num_layers': {'values': [20, 25, 30]},
        # constants
        'epochs': {'value': 500},
        'dropout': {'value': 0.0},
     }
}

# setup config
default_config = {
    'lr_start': 0.00001,
    'lr_max': 0.005,
    'batch_size': 512,
    'warmup_steps': 1000,
    'epochs': 500,
    'hidden_dim': 512,
    'dropout': 0.0,
    'num_layers': 25,
}


def main(run_config=None):
    csv_path = r"C:\Users\mktha\Documents\projects\felix\data\RawDataMagExtended(newdataset).csv"

    df_mag = pd.read_csv(csv_path, header=0, index_col=0)

    csv_path = r"C:\Users\mktha\Documents\projects\felix\data\mag30000.csv"

    df_mag_2 = pd.read_csv(csv_path, header=0, index_col=0)

    csv_path = r"C:\Users\mktha\Documents\projects\felix\data\mag100000.csv"

    df_mag_3 = pd.read_csv(csv_path, header=0, index_col=0)

    # concat the two dataframes
    df_mag = pd.concat([df_mag, df_mag_2, df_mag_3], axis=0)

    # get the headers
    headers = df_mag.columns.values.tolist()

    ds_path = r"C:\Users\mktha\Documents\projects\felix\data\extended500000.mat"

    f = scipy.io.loadmat(ds_path)
    df = pd.DataFrame(f['extended500000mag'])
    # drop the first column
    df = df.drop([0], axis=1)
    # rename the columns
    df.columns = headers
    # concat the two dataframes vertically
    df_mag = pd.concat([df_mag, df], axis=0)


    df1 = df_mag
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




    # root_path = Path(r'C:\Users\mktha\Documents\projects\felix')

    # model_dir = root_path / 'models'
    wandb.init(project='felix')
    # get run name
    run_name = wandb.run.name
    # get run dir
    run_dir = wandb.run.dir

    wandb.run.log_code('.')
    if run_config is None:
        config = wandb.config
    else:
        config = run_config

    # %%

    num_inputs = X_train.shape[1]
    num_outputs = y_train.shape[1]

    input_layer = tf.keras.Input(shape=(num_inputs,))
    x = layers.Dense(config['hidden_dim'], activation='relu')(input_layer)
    x = layers.LayerNormalization()(x)
    if config['dropout'] > 0:
        x = layers.Dropout(config['dropout'])(x)
    for layer in range(config['num_layers'] - 1):
        residual = x
        x = layers.Dense(config['hidden_dim'], activation='relu')(x)
        x = layers.LayerNormalization()(x)
        if config['dropout'] > 0:
            x = layers.Dropout(config['dropout'])(x)
        # x = layers.Dense(hidden_dim, activation='relu')(x)
        # x = layers.LayerNormalization()(x)
        # x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([x, residual])
    output_layer = layers.Dense(num_outputs)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    schedule = WarmUpCosine(lr_start=config['lr_start'], lr_max=config['lr_max'], warmup_steps=config['warmup_steps'],
                            total_steps=config['epochs']*len(X_train)//config['batch_size'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'../models/cp_{run_name}.h5', monitor='val_loss',
                                                     save_best_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    wandb_callback = wandb.keras.WandbCallback(monitor="val_loss",
                                                       log_weights=True, save_model=False)
    model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], validation_data=(X_test, y_test),
              callbacks=[lr_callback, cp_callback, es_callback, wandb_callback])


    # # predict
    # model = tf.keras.models.load_model('../models/best_model.h5')
    # y_pred = model.predict(X_test)
    # y_pred = values_scaler.inverse_transform(y_pred)
    # y_test = values_scaler.inverse_transform(y_test)
    #
    #
    # # calculate abs error along axis 1
    # error_abs = np.abs(y_pred - y_test)
    # square_error = np.square(y_pred - y_test)
    # mse = np.mean(np.square(error_abs))
    # mae = np.mean(error_abs)
    # print("MSE: ", mse)
    # print("MAE: ", mae)

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='felix')
wandb.agent(sweep_id, main, count=20)
# main(default_config)

