import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from keras import layers
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
        'batch_size': {'values': [512, 1024, 2048]},
        'lr_start': {'max': 0.0001, 'min': 0.00001},
        'lr_max': {'max': 0.01, 'min': 0.005},
        'warmup_steps': {'values': [500, 1000, 1500]},
        'hidden_dim': {'values': [512, 1024, 2048]},
        # 'dropout': {'values': [0.0, 0.1, 0.2, 0.3]},
        'num_layers': {'values': [20, 25, 30]},
        # constants
        'epochs': {'value': 500},
        'dropout': {'value': 0.0},
     }
}

# setup config
default_config = {
    'lr_start': 0.00008,
    'lr_max': 0.005,
    'batch_size': 1024,
    'warmup_steps': 1000,
    'epochs': 500,
    'hidden_dim': 512,
    'dropout': 0.0,
    'num_layers': 20,
}


def main(run_config=None):
    csv_path = r"C:\Users\mktha\Documents\projects\metamaterials_design\data\extended300000mag.csv"
    df_mag_1 = pd.read_csv(csv_path, header=0, index_col=0)
    csv_path = r"C:\Users\mktha\Documents\projects\metamaterials_design\data\extended350000mag.csv"
    df_mag_2 = pd.read_csv(csv_path, header=0, index_col=0)
    df_mag = pd.concat([df_mag_1, df_mag_2], axis=0)

    csv_path = r"C:\Users\mktha\Documents\projects\metamaterials_design\data\extended300000ph.csv"
    df_phase_1 = pd.read_csv(csv_path, header=0, index_col=0)
    csv_path = r"C:\Users\mktha\Documents\projects\metamaterials_design\data\extended350000ph.csv"
    df_phase_2 = pd.read_csv(csv_path, header=0, index_col=0)
    df_phase = pd.concat([df_phase_1, df_phase_2], axis=0)

    param1 = df_mag['eps1 '].values
    param2 = df_mag['eps2 '].values
    param3 = df_mag['eps3 '].values
    param4 = df_mag['eps4 '].values
    param5 = df_mag['t1 [mm]'].values
    param6 = df_mag['t2 [mm]'].values
    param7 = df_mag['t3 [mm]'].values
    param8 = df_mag['t4 [mm]'].values
    # get the columns from 9 to the end
    mag_values = df_mag.iloc[:, 8:].values
    phase_values = df_phase.iloc[:, 8:].values
    real_values = mag_values * np.cos(np.deg2rad(phase_values))
    imag_values = mag_values * np.sin(np.deg2rad(phase_values))

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
    real_scaler = preprocessing.StandardScaler()
    real_values = real_scaler.fit_transform(real_values)
    imag_scaler = preprocessing.StandardScaler()
    imag_values = imag_scaler.fit_transform(imag_values)

    # concat real and imag
    values = np.column_stack((real_values, imag_values))

    inputs = np.column_stack((param1, param2, param3, param4, param5, param6, param7, param8))
    # split data
    X_train, X_test, y_train, y_test = train_test_split(inputs, values, test_size=0.2, random_state=42)




    # root_path = Path(r'C:\Users\mktha\Documents\projects\felix')

    # model_dir = root_path / 'models'
    wandb.init(project='metamaterials_toy_mdn')
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
    # lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'../models/cp_{run_name}.h5', monitor='val_loss',
                                                     save_best_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    wandb_callback = wandb.keras.WandbCallback(monitor="val_loss",
                                                       log_weights=True, save_model=False)
    model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], validation_data=(X_test, y_test),
              callbacks=[cp_callback, es_callback, wandb_callback])


    # # predict
    model = tf.keras.models.load_model('../models/best_model.h5')
    y_pred = model.predict(X_test)
    # split real and imag
    y_pred = np.split(y_pred, 2, axis=1)
    real_pred = y_pred[0]
    imag_pred = y_pred[1]

    # inverse standardize
    real_pred = real_scaler.inverse_transform(real_pred)
    imag_pred = imag_scaler.inverse_transform(imag_pred)
    y_pred = np.column_stack((real_pred, imag_pred))
    
    
    # calculate abs error along axis 1
    error_abs = np.abs(y_pred - y_test)
    square_error = np.square(y_pred - y_test)
    mse = np.mean(np.square(error_abs))
    mae = np.mean(error_abs)
    print("MSE: ", mse)
    print("MAE: ", mae)
    wandb.log({'mse': mse, 'mae': mae})

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='metamaterials_toy_mdn')
# wandb.agent(sweep_id, main, count=20)
main(default_config)

