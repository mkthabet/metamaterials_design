import tensorflow as tf
from tensorflow_probability import distributions as tfd

def get_mdn_loss(output_dim, num_components):
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
    return mdn_loss
