import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

def convert_alpha(alpha):
    one_over_sqrt_alpha  = tf.map_fn(lambda x:  1/(tf.math.sqrt(x)), alpha, dtype=tf.float64)
    return one_over_sqrt_alpha

a_0 = tf.constant(1, dtype=tf.float64)
b_0 = tf.constant(1, dtype=tf.float64)
c_0 = tf.constant(1, dtype=tf.float64)
d_0 = tf.constant(1, dtype=tf.float64)

def alpha_prior_():
    return tfd.Gamma(a_0, b_0)

def tau_prior_():
    return tfd.InverseGamma(c_0, d_0)

def w_prior_(sigma, one_over_sqrt_alpha):
    return tfd.Normal(0,tf.math.multiply(sigma, one_over_sqrt_alpha))

def sep_params(matrix, features):
    w = matrix[:features]
    tau = matrix[features+1]
    alpha = matrix[features+1:]
    return w, tau, alpha

def joint_log_prob(y, x, params):
    features = x.shape[0]
    # Must check
    # input current estimates of parameters, data,
    # return log joint distribution
    w, tau, alpha = sep_params(params, features)
    alpha_prior = alpha_prior_()
    one_over_sqrt_alpha = convert_alpha(alpha)
    tau_prior = tau_prior_()
    sigma = tf.math.sqrt(tau)
    w_prior = w_prior_(sigma, one_over_sqrt_alpha)
    log_likelihood = tfd.Normal(tf.linalg.matvec(x, w, transpose_a=True), sigma).log_prob(y)
    sum_alpha_prior = tf.reduce_sum(alpha)
    sum_is = tf.reduce_sum(
        log_likelihood) + tf.reduce_sum(w_prior.log_prob(w)) + tau_prior.log_prob(tau) + tf.reduce_sum(alpha_prior.log_prob(alpha)) 
    return sum_is

def some_kind_of_loss(y, x, w):
    y_pred = tf.linalg.matvec(x, w, transpose_a=True)
    return tf.reduce_sum(tf.math.abs(tf.math.subtract(y, y_pred)))
    
def log_likelihood(y, x, params):
    features = x.shape[0]
    w, tau, alpha = sep_params(params, features)
    alpha_prior = alpha_prior_()
    one_over_sqrt_alpha = convert_alpha(alpha)
    tau_prior = tau_prior_()
    sigma = tf.math.sqrt(tau)
    w_prior = w_prior_(sigma, one_over_sqrt_alpha)
    log_likelihood_ = tf.reduce_sum(tfd.Normal(
        tf.linalg.matvec(x, w, transpose_a=True), sigma).log_prob(y))
    return log_likelihood_

def return_initial_state(features):
    alpha = alpha_prior_().sample([features])
    tau = tau_prior_().sample()
    sigma = tf.math.sqrt(tau)
    tau = tf.reshape(tau, [1])
    one_over_sqrt_alpha = convert_alpha(alpha)
    w = w_prior_(sigma, one_over_sqrt_alpha).sample()
    return tf.concat([w, tau, alpha],0)
     
