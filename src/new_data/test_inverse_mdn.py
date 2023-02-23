import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
from lib.schedulers import WarmUpCosine
from lib.losses import get_mdn_loss

sample = False
sample_size = 5

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
response_train, response_test, params_train, params_test = \
    train_test_split(values, params, test_size=0.2, random_state=42)


response_dim = response_train.shape[1]
input_dim = response_dim
num_params = params_train.shape[1]
output_dim = num_params

# load the models
forward_model_path = r"C:\Users\mktha\Documents\projects\felix\models\cp_restful-sweep-2.h5"
inverse_model_path = r"C:\Users\mktha\Documents\projects\felix\models\cp_twilight-sweep-15.h5"
# predict
num_components = 64
forward_model = tf.keras.models.load_model(forward_model_path, custom_objects={'WarmUpCosine': WarmUpCosine})
inverse_model = tf.keras.models.load_model(inverse_model_path, custom_objects={'mdn_loss': get_mdn_loss(output_dim, num_components),
                                                       'WarmUpCosine': WarmUpCosine})


preds = inverse_model.predict(response_test)
mus = preds[:, :num_params * num_components]
sigmas = preds[:, num_params * num_components:2 * num_params * num_components]


pis = preds[:, 2 * num_params * num_components:]
mus = np.reshape(mus, [-1, num_components, num_params])
sigmas = np.reshape(sigmas, [-1, num_components, num_params])

# inverse transform mus and sigmas
mus = params_scaler.inverse_transform(mus.reshape(-1, num_params)).reshape(-1, num_components, num_params)
sigmas = sigmas.reshape(-1, num_params) * params_scaler.scale_
sigmas = sigmas.reshape(-1, num_components, num_params)
pis = np.reshape(pis, [-1, num_components])

mus_transformed = mus
sigmas_transformed = sigmas



_pis = np.expand_dims(pis, -1)
mdn_mean = np.sum(_pis * mus, axis=1, keepdims=False)
mdn_mean_sq = np.power(mdn_mean, 2)
weighted_sq_mean = np.sum(_pis * np.power(mus, 2), axis=1, keepdims=False)
modal_variance = weighted_sq_mean - mdn_mean_sq
relative_modal_variance = modal_variance / mdn_mean
mean_modal_variance = np.mean(modal_variance, axis=0)
mean_relative_modal_variance = np.mean(relative_modal_variance, axis=0)

print(f'mean modal variance: {mean_modal_variance}')
print(f'mean relative modal variance: {mean_relative_modal_variance}')

# inverse_model_vanilla = tf.keras.models.load_model(inverse_model_vanilla_path)
# preds_vanilla = inverse_model_vanilla.predict(response_test)

rng = np.random.default_rng()
response_gt = values_scaler.inverse_transform(response_test)
# response_preds_vanilla = forward_model.predict(preds_vailla)
# inverse transform preds
# preds_vanilla_param1 = preds_vanilla[:, 0]
# preds_vanilla_param2 = preds_vanilla[:, 1]
# preds_vanilla_param3 = preds_vanilla[:, 2]
# preds_vanilla_param1 = param1_scaler.inverse_transform(preds_vanilla_param1.reshape(-1, 1))
# preds_vanilla_param2 = param2_scaler.inverse_transform(preds_vanilla_param2.reshape(-1, 1))
# preds_vanilla_param3 = param3_scaler.inverse_transform(preds_vanilla_param3.reshape(-1, 1))
# response_preds_vanilla = values_scaler.inverse_transform(response_preds_vanilla)
# unstack parameters in y_test
params_gt = params_scaler.inverse_transform(params_test)
for i in range(preds.shape[0]):
    if sample:
        # sample from the mixture
        idx = rng.choice(num_components, size=sample_size, p=pis[i])
        mu = mus_transformed[i, idx, :]
        sigma = sigmas_transformed[i, idx, :]
        sample_pred = np.random.normal(loc=mu, scale=sigma)
        pis_filtered = pis[i, idx]
    else:
        sample_pred = mus_transformed[i, :, :]
        # arrange sample pred in descending according to pis[i]
        idx = np.argsort(pis[i])[::-1]
        sample_pred = sample_pred[idx, :]
        pis[i] = pis[i][idx]
        # filter out values of pis below 0.1
        idx = pis[i] > 0.05
        sample_pred = sample_pred[idx, :]
        pis_filtered = pis[i][idx]
        sample_size = len(pis_filtered)
    # inverse transform zero
    # transformed_zero = params_scaler.transform(np.zeros((1, 1)))
    # # clip sample_pred to zero on param1
    # sample_pred[:, 0] = np.clip(sample_pred[:, 0], transformed_zero, np.inf)
    sample_pred = params_scaler.transform(sample_pred)
    response_pred = forward_model.predict(sample_pred)
    # unstandardize
    response_pred = values_scaler.inverse_transform(response_pred)
    params_pred = params_scaler.inverse_transform(sample_pred)
    # unstack the 8 parameters
    param1_pred = params_pred[:, 0]
    param2_pred = params_pred[:, 1]
    param3_pred = params_pred[:, 2]
    param4_pred = params_pred[:, 3]
    param5_pred = params_pred[:, 4]
    param6_pred = params_pred[:, 5]
    param7_pred = params_pred[:, 6]
    param8_pred = params_pred[:, 7]
    param1_gt = params_gt[i, 0]
    param2_gt = params_gt[i, 1]
    param3_gt = params_gt[i, 2]
    param4_gt = params_gt[i, 3]
    param5_gt = params_gt[i, 4]
    param6_gt = params_gt[i, 5]
    param7_gt = params_gt[i, 6]
    param8_gt = params_gt[i, 7]

    param1_gt_str = f'p1: {param1_gt:.2f}'
    param2_gt_str = f'p2: {param2_gt:.2f}'
    param3_gt_str = f'p3: {param3_gt:.2f}'
    param4_gt_str = f'p4: {param4_gt:.2f}'
    param5_gt_str = f'p5: {param5_gt:.2f}'
    param6_gt_str = f'p6: {param6_gt:.2f}'
    param7_gt_str = f'p7: {param7_gt:.2f}'
    param8_gt_str = f'p8: {param8_gt:.2f}'
    print(f'modal mean: {mdn_mean[i, :]}, modal variance: {modal_variance[i, :]},'
          f' relative modal variance: {relative_modal_variance[i, :]}')
    plt.plot(response_gt[i], label=f'input response at {param1_gt_str}, {param2_gt_str}, {param3_gt_str},'
                                   f' {param4_gt_str}, {param5_gt_str}, {param6_gt_str}, {param7_gt_str}, {param8_gt_str}')

    for sample_num in range(sample_size):
        param1_str = f'p1: {param1_pred[sample_num]:.2f}'
        param2_str = f'p2: {param2_pred[sample_num]:.2f}'
        param3_str = f'p3: {param3_pred[sample_num]:.2f}'
        param4_str = f'p4: {param4_pred[sample_num]:.2f}'
        param5_str = f'p5: {param5_pred[sample_num]:.2f}'
        param6_str = f'p6: {param6_pred[sample_num]:.2f}'
        param7_str = f'p7: {param7_pred[sample_num]:.2f}'
        param8_str = f'p8: {param8_pred[sample_num]:.2f}'
        pi_str = f'pi: {pis_filtered[sample_num]:.2f}' if not sample else ''
        label_str = f'pred response for {param1_str}, {param2_str}, {param3_str}, {param4_str}, {param5_str},' \
                    f' {param6_str}, {param7_str}, {param8_str}'
        if not sample:
            label_str = label_str + f', {pi_str}'
        plt.plot(response_pred[sample_num],
                 label=label_str, linestyle='--')
    # vanilla_param1_str = f'p1: {preds_vanilla_param1[i][0]:.2f}'
    # vanilla_param2_str = f'p2: {preds_vanilla_param2[i][0]:.2f}'
    # vanilla_param3_str = f'p3: {preds_vanilla_param3[i][0]:.2f}'
    # vanilla_label_str = f'vanilla pred response for {vanilla_param1_str}, {vanilla_param2_str}, {vanilla_param3_str}'
    # plt.plot(response_preds_vanilla[i], label=vanilla_label_str, linestyle=':', color='black')
    plt.title('Predicted response for samples of predicted parameters')
    plt.legend()
    plt.show()