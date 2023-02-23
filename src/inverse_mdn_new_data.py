import scipy.io
import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import wandb
from lib.schedulers import WarmUpCosine
from lib.losses import get_mdn_loss
from lib.callbacks import LRLogger

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
    {
        'batch_size': {'values': [1024, 2048, 4096]},
        'lr_start': {'max': 0.000001, 'min': 0.0000001},
        'lr_max': {'max': 0.0005, 'min': 0.000005},
        'warmup_steps': {'values': [4000, 5000, 6000, 7000]},
        'hidden_dim': {'values': [512, 1024, 2048]},
        'dropout': {'values': [0.0, 0.1, 0.2]},
        'num_layers': {'values': [20, 25, 30]},
        'num_components': {'values': [48, 64, 80]},
        # constants
        'epochs': {'value': 700},
        # 'dropout': {'value': 0.0},
     }
}

default_config = {
    'lr_start': 0.000001,
    'lr_max': 0.00001,
    'batch_size': 1024,
    'warmup_steps': 3000,
    'epochs': 500,
    'hidden_dim': 1024,
    'dropout': 0.2,
    'num_layers': 25,
    'num_components': 64,
}



def train_MDN(run_config=None):
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

    params = np.column_stack((param1, param2, param3, param4, param5, param6, param7, param8))

    # standardize
    params_scaler = preprocessing.StandardScaler()
    params = params_scaler.fit_transform(params)
    values_scaler = preprocessing.StandardScaler()
    values = values_scaler.fit_transform(values)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(values, params,
                                                        test_size=0.2, random_state=42)
    wandb.init(project='felix_inverse_mdn')
    # get run name
    run_name = wandb.run.name
    # get run dir
    run_dir = wandb.run.dir

    wandb.run.log_code('.')
    if run_config is None:
        run_config = wandb.config
    else:
        wandb.config = run_config

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    num_components = run_config['num_components']
    input_layer = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(run_config['hidden_dim'], activation='relu')(input_layer)
    x = layers.LayerNormalization()(x)
    if run_config['dropout'] > 0:
        x = layers.Dropout(run_config['dropout'])(x)
    for layer in range(run_config['num_layers'] - 1):
        residual = x
        x = layers.Dense(run_config['hidden_dim'], activation='relu')(x)
        x = layers.LayerNormalization()(x)
        if run_config['dropout'] > 0:
            x = layers.Dropout(run_config['dropout'])(x)
        x = layers.Add()([x, residual])
    mu_output = layers.Dense(output_dim * run_config['num_components'], activation='linear', name='mu_output')(x)
    sigma_output = layers.Dense(output_dim * run_config['num_components'], activation='softplus', name='sigma_output')(x)
    pi_output = layers.Dense(num_components, activation='softmax', name='pi_output')(x)
    output = layers.Concatenate(axis=-1, name='output')([mu_output, sigma_output, pi_output])
    model = tf.keras.Model(inputs=input_layer, outputs=output)

    schedule = WarmUpCosine(lr_start=run_config['lr_start'], lr_max=run_config['lr_max'], warmup_steps=run_config['warmup_steps'],
                            total_steps=run_config['epochs'] * len(X_train) // run_config['batch_size'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    model.compile(optimizer=optimizer, loss=get_mdn_loss(output_dim, num_components))
    model.summary()
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001,
                                                       verbose=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'../models/cp_{run_name}.h5', monitor='val_loss',
                                                     save_best_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    lr_logger_callback = LRLogger(optimizer=optimizer)
    wandb_callback = wandb.keras.WandbCallback(monitor="val_loss",
                                               log_weights=True, save_model=False)
    model.fit(X_train, y_train, epochs=run_config['epochs'], batch_size=run_config['batch_size'],
              validation_data=(X_test, y_test),
              callbacks=[lr_callback, cp_callback, es_callback, wandb_callback, lr_logger_callback])

    model = tf.keras.models.load_model(f'../models/cp_{run_name}.h5',
                                       custom_objects={'mdn_loss': get_mdn_loss(output_dim, num_components),
                                                       'WarmUpCosine': WarmUpCosine})
    preds = model.predict(X_test)
    mus = preds[:, :output_dim * num_components]
    sigmas = preds[:, output_dim * num_components:2 * output_dim * num_components]
    pis = preds[:, 2 * output_dim * num_components:]
    mus = np.reshape(mus, [-1, num_components, output_dim])
    sigmas = np.reshape(sigmas, [-1, num_components, output_dim])
    pis = np.reshape(pis, [-1, num_components])
    # get best component
    y_pred = np.zeros((y_test.shape[0], output_dim))
    best_component = np.argmax(pis, axis=-1)
    # for i in range(len(best_component)):
    #     y_pred[i] = mus[i][best_component[i]]
    for i, y in enumerate(y_test):
        # find the component that minimizes the distance to the true value
        dist = np.linalg.norm(mus[i] - y, axis=-1)
        best_component = np.argmin(dist)
        y_pred[i] = mus[i][best_component]

    # unstandardize
    params_pred = params_scaler.inverse_transform(y_pred)
    params_test = params_scaler.inverse_transform(y_test)

    # calculate abs error for each parameter
    params_abs_error = np.abs(params_pred - params_test)
    # calculate mean abs error for each parameter
    params_mae = np.mean(params_abs_error, axis=0)
    # calulate mse for each parameter
    params_mse = np.mean(np.square(params_pred - params_test), axis=0)
    # clauculate percentage error for each parameter
    params_percentage_error = np.mean(np.abs(params_pred - params_test) / params_test, axis=0) * 100
    # print results
    print('MAE: ', params_mae)
    print('MSE: ', params_mse)
    print('Percentage Error: ', params_percentage_error)
    # log results
    wandb.log({'MAE': params_mae.mean(), 'MSE': params_mse.mean(), 'Percentage Error': params_percentage_error.mean()})

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='felix_inverse_mdn')
    wandb.agent(sweep_id, train_MDN, count=30)
    # train_MDN(default_config)

