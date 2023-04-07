import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import wandb
from lib.schedulers import WarmUpCosine

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': '',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters':
    {
        'batch_size': {'values': [1024, 2048, 4096]},
        'lr_start': {'max': 0.0001, 'min': 0.00001},
        'lr_max': {'max': 0.01, 'min': 0.001},
        'warmup_steps': {'values': [500, 1000, 1500]},
        'hidden_dim': {'values': [256, 512, 1024]},
        'dropout': {'values': [0.0, 0.1, 0.2, 0.3]},
        'num_layers': {'values': [15, 20, 25]},
        # constants
        'epochs': {'value': 500},
        # 'dropout': {'value': 0.0},
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
    csv_path = r"data/Txy_mag(all).csv"
    df_mag = pd.read_csv(csv_path)
    # csv_path = r"C:\Users\mktha\Documents\projects\felix\data\extended350000mag.csv"
    # df_mag_2 = pd.read_csv(csv_path, header=0, index_col=0)
    # df_mag = pd.concat([df_mag_1, df_mag_2], axis=0)

    params = df_mag.iloc[:, 0:8].values
    # get the columns from 9 to the end
    mag_values = df_mag.iloc[:, 8:].values
    # take the third quarter of the data
    # mag_values = mag_values[:, 250:]

    # standardize
    param_scaler = preprocessing.StandardScaler()
    params = param_scaler.fit_transform(params)
    mag_scaler = preprocessing.StandardScaler()
    mag_values = mag_scaler.fit_transform(mag_values)

    inputs = params
    # split data
    X_train, X_test, y_train, y_test = train_test_split(inputs, mag_values, test_size=0.1, random_state=42)




    # root_path = Path(r'C:\Users\mktha\Documents\projects\felix')

    # model_dir = root_path / 'models'
    wandb.init(project='metamaterials_task2_1')
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
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'models/cp_{run_name}.h5', monitor='val_loss',
                                                     save_best_only=True, verbose=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    wandb_callback = wandb.keras.WandbCallback(monitor="val_loss",
                                                       log_weights=False, save_model=False)
    model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], validation_data=(X_test, y_test),
              callbacks=[cp_callback, es_callback, wandb_callback])


    # # predict
    model = tf.keras.models.load_model(f'models/cp_{run_name}.h5', custom_objects={'WarmUpCosine': WarmUpCosine})
    y_pred = model.predict(X_test)

    # inverse standardize
    y_pred = mag_scaler.inverse_transform(y_pred)
    y_test = mag_scaler.inverse_transform(y_test)
    
    # calculate abs error along axis 1
    error_abs = np.abs(y_pred - y_test)
    mse = np.mean(np.square(error_abs))
    mae = np.mean(error_abs)
    print("MSE: ", mse)
    print("MAE: ", mae)
    wandb.log({'mse': mse, 'mae': mae})

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='metamaterials_forward_actual_data')
wandb.agent(sweep_id, main, count=30)
# main(default_config)

