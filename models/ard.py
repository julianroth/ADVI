import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

def convert_alpha(alpha):
    print("converting alpha")
    one_over_sqrt_alpha  = tf.map_fn(lambda x:  1/(tf.math.sqrt(x)), alpha, dtype=tf.float64)
    print("one_over_sqrting alpha", one_over_sqrt_alpha.shape)
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
    print("w issss", w.shape)
    print("tau isss", tau.shape)
    print("alpha iss", alpha.shape)
    return w, tau, alpha

def joint_log_prob(y, x, params):
    features = x.shape[0]
    print("features!!!", features)
    # Must check
    # input current estimates of parameters, data,
    # return log joint distribution
    w, tau, alpha = sep_params(params, features)
    print("j check1")
    alpha_prior = alpha_prior_()
    print("j check2")   
    one_over_sqrt_alpha = convert_alpha(alpha)
    tau_prior = tau_prior_()
    print("j check3")   
    sigma = tf.math.sqrt(tau)
    print("j check4")   
    w_prior = w_prior_(sigma, one_over_sqrt_alpha)
    print("j check5")   
    likelihood = tfd.Normal(tf.linalg.matvec(x, w, transpose_a=True), sigma).log_prob(y)
    sum_alpha_prior = tf.reduce_sum(alpha)
    print("join log called")
    sum_is = tf.reduce_sum(likelihood) + w_prior.log_prob(w) + tau_prior.log_prob(tau) + tf.reduce_sum(alpha_prior.log_prob(alpha)) 
    print("sum complete")
    return sum_is

def return_initial_state(features):
    print("check 7")
    alpha = alpha_prior_().sample([features])
    print("alpha is",alpha.shape)
    tau = tau_prior_().sample()
    sigma = tf.math.sqrt(tau)
    print("tau is", tau)
    tau = tf.reshape(tau, [1])
    print("sigma", sigma)
    one_over_sqrt_alpha = convert_alpha(alpha)
    print("check 10")
    w = w_prior_(sigma, one_over_sqrt_alpha).sample()

    print("concat shape", tf.concat([w, tau, alpha],0).shape )
    return tf.concat([w, tau, alpha],0)
     
