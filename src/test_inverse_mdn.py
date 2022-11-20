import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sample = True
sample_size = 5

csv_path = r"C:\Users\mktha\Documents\projects\felix\data\data.csv"
forward_model_path = '../models/forward_model_v1.0.h5'
inverse_model_path = '../models/mdn_v1.1.h5'

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
response_dim = X_train.shape[1]
input_dim = response_dim
num_params = y_train.shape[1]
output_dim = num_params
num_components = 5

@tf.keras.utils.register_keras_serializable()
def mdn_loss(y_true, y_pred):
    mu = y_pred[:, :output_dim * num_components]
    sigma = y_pred[:, output_dim * num_components:2 * output_dim * num_components]
    pi = y_pred[:, 2 * output_dim * num_components:]
    mu = tf.reshape(mu, [-1, num_components, output_dim])
    sigma = tf.reshape(sigma, [-1, num_components, output_dim])
    pi = tf.reshape(pi, [-1, num_components])
    y_true = tf.reshape(y_true, [-1, 1, output_dim])
    y_true = tf.tile(y_true, [1, num_components, 1])
    dist = tfd.Normal(loc=mu, scale=sigma)
    log_prob = dist.log_prob(y_true)
    log_prob = tf.reduce_sum(log_prob, axis=-1)
    log_prob = log_prob + tf.math.log(pi)
    log_prob = tf.reduce_logsumexp(log_prob, axis=-1)
    loss = -tf.reduce_mean(log_prob)
    return loss
# predict
forward_model = tf.keras.models.load_model(forward_model_path)
inverse_model = tf.keras.models.load_model(inverse_model_path)
preds = inverse_model.predict(X_test)
mus = preds[:, :num_params * num_components]
sigmas = preds[:, num_params * num_components:2 * num_params * num_components]
pis = preds[:, 2 * num_params * num_components:]
mus = np.reshape(mus, [-1, num_components, num_params])
sigmas = np.reshape(sigmas, [-1, num_components, num_params])
pis = np.reshape(pis, [-1, num_components])

mus_transformed = mus
sigmas_transformed = sigmas
# inverse transform mus and sigmas
mus_param1 = mus[:, :, 0]
mus_param2 = mus[:, :, 1]
mus_param3 = mus[:, :, 2]
sigmas_param1 = sigmas[:, :, 0]
sigmas_param2 = sigmas[:, :, 1]
sigmas_param3 = sigmas[:, :, 2]
mus_param1 = param1_scaler.inverse_transform(mus_param1)
mus_param2 = param2_scaler.inverse_transform(mus_param2)
mus_param3 = param3_scaler.inverse_transform(mus_param3)
sigmas_param1 = sigmas_param1 * param1_scaler.scale_
sigmas_param2 = sigmas_param2 * param2_scaler.scale_
sigmas_param3 = sigmas_param3 * param3_scaler.scale_
mus = np.stack((mus_param1, mus_param2, mus_param3), axis=2)
sigmas = np.stack((sigmas_param1, sigmas_param2, sigmas_param3), axis=2)


_pis = np.expand_dims(pis, -1)
mdn_mean = np.sum(_pis * mus, axis=1, keepdims=False)
mdn_mean_sq = np.power(mdn_mean, 2)
weighted_sq_mean = np.sum(_pis * np.power(mus, 2), axis=1, keepdims=False)
modal_variance = weighted_sq_mean - mdn_mean_sq
relative_modal_variance = modal_variance / mdn_mean
mean_modal_variance = np.mean(modal_variance, axis=0)
mean_relative_modal_variance = np.mean(relative_modal_variance, axis=0)

print(f'mean modal variance: param1: {mean_modal_variance[0]}, param2: {mean_modal_variance[1]}, param3: {mean_modal_variance[2]}')
print(f'mean relative modal variance: param1: {mean_relative_modal_variance[0]}, param2: {mean_relative_modal_variance[1]}, param3: {mean_relative_modal_variance[2]}')

rng = np.random.default_rng()
response_gt = values_scaler.inverse_transform(X_test)
# unstack parameters in y_test
param1_gt = param1_scaler.inverse_transform(y_test[:, 0].reshape(-1, 1))
param2_gt = param2_scaler.inverse_transform(y_test[:, 1].reshape(-1, 1))
param3_gt = param3_scaler.inverse_transform(y_test[:, 2].reshape(-1, 1))
if sample:
    for i in range(preds.shape[0]):
        #sample from the mixture
        idx = rng.choice(num_components, size=sample_size, p=pis[i])
        mu = mus_transformed[i, idx, :]
        sigma = sigmas_transformed[i, idx, :]
        sample_pred = np.random.normal(loc=mu, scale=sigma)
        response_pred = forward_model.predict(sample_pred)
        # unstack params
        param1_pred = sample_pred[:, 0]
        param2_pred = sample_pred[:, 1]
        param3_pred = sample_pred[:, 2]
        # unstandardize
        response_pred = values_scaler.inverse_transform(response_pred)
        param1_pred = param1_scaler.inverse_transform(param1_pred.reshape(-1, 1))
        param2_pred = param2_scaler.inverse_transform(param2_pred.reshape(-1, 1))
        param3_pred = param3_scaler.inverse_transform(param3_pred.reshape(-1, 1))
        param1_pred = param1_pred.reshape(-1, 1)
        param2_pred = param2_pred.reshape(-1, 1)
        param3_pred = param3_pred.reshape(-1, 1)
        param1_gt_str = f'p1: {param1_gt[i, 0]:.2f}'
        param2_gt_str = f'p2: {param2_gt[i, 0]:.2f}'
        param3_gt_str = f'p3: {param3_gt[i, 0]:.2f}'
        print(f'modal mean: {mdn_mean[i, :]}, modal variance: {modal_variance[i, :]}, relative modal variance: {relative_modal_variance[i, :]}')
        plt.plot(response_gt[i], label=f'input response at {param1_gt_str}, {param2_gt_str}, {param3_gt_str}')

        for sample_num in range(sample_size):
            param1_str = f'p1: {param1_pred[sample_num][0]:.2f}'
            param2_str = f'p2: {param2_pred[sample_num][0]:.2f}'
            param3_str = f'p3: {param3_pred[sample_num][0]:.2f}'
            plt.plot(response_pred[sample_num],
                     label=f'pred response for {param1_str}, {param2_str}, {param3_str}', linestyle='--')

        plt.title('Predicted response for samples of predicted parameters')
        plt.legend()
        plt.show()
else:
    # get best component
    params_pred = np.zeros((y_test.shape[0], num_params))
    best_component = np.argmax(pis, axis=-1)
    for i in range(len(best_component)):
        params_pred[i] = mus[i][best_component[i]]

    response_pred = forward_model.predict(params_pred)
    #unstack params
    param1_pred = params_pred[:, 0]
    param2_pred = params_pred[:, 1]
    param3_pred = params_pred[:, 2]
    # unstandardize
    response_pred = values_scaler.inverse_transform(response_pred)
    response_gt = values_scaler.inverse_transform(X_test)
    param1_pred = param1_scaler.inverse_transform(param1_pred.reshape(-1, 1))
    param2_pred = param2_scaler.inverse_transform(param2_pred.reshape(-1, 1))
    param3_pred = param3_scaler.inverse_transform(param3_pred.reshape(-1, 1))
    for i in range(len(response_pred)):
        # plot response
        plt.plot(response_pred[i], label='inverse model')
        plt.plot(response_gt[i], label='forward model')
        plt.legend()
        plt.title(f'param1: {param1_pred[i]}, param2: {param2_pred[i]}, param3: {param3_pred[i]}')
        plt.show()