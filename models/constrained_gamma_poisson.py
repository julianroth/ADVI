import tensorflow as tf
import tensorflow_probability as tfp
from data import frey_face

tfd = tfp.distributions

# This implements the constrained gamma poisson distribution from 3.2
data = frey_face.load_data()

# Values from paper
K = 10
U = 28
I = 20
a_0 = tf.constant(1, dtype=tf.float64)
b_0 = tf.constant(1, dtype=tf.float64)
c_0 = tf.constant(1, dtype=tf.float64)
d_0 = tf.constant(1, dtype=tf.float64)

def theta_prior_distribution():
    return tfd.Gamma(a_0, b_0)

def beta_prior_distribution():
    return tfd.Gamma(c_0, d_0)

def stddev_theta(tile=True):
    theta_dist = theta_prior_distribution()
    stddev = theta_dist.stddev()
    if tile:
        stddev = tf.reshape(stddev, [1, 1])
        stddev = tf.tile(stddev, tf.constant([U, K], dtype=tf.int64))
    return stddev

def stddev_beta(tile=True):
    beta_dist = beta_prior_distribution()
    stddev = beta_dist.stddev()
    if tile:
        stddev = tf.reshape(stddev, [1, 1])
        stddev = tf.tile(stddev, tf.constant([K, I], dtype=tf.int64))
    return stddev

def initial_state():
    return [theta_prior_distribution().sample([U, K]), beta_prior_distribution().sample([K, I])]

def step_size():
    return [stddev_theta(), stddev_beta()]

# don't need this for nuts
def log_prior(theta, beta):
    theta_dist = tfd.Gamma(a_0, b_0)
    theta_prob = theta_dist.log_prob(theta)
    beta_dist = tfd.Gamma(c_0, d_0)
    beta_prob = beta_dist.log_prob(beta)
    return tf.math.reduce_sum(theta_prob) + tf.math.reduce_sum(beta_prob)

def log_likelihood(theta, beta):
    theta = tf.reshape(theta, [U, K])
    beta = tf.reshape(beta, [K, I])
    params = tf.linalg.matmul(theta, beta)
    params = tf.reshape(params, [-1])
    posterior_dist = tfd.Poisson(params)
    log_prob = posterior_dist.log_prob(data)
    #log_prob = tf.math.reduce_mean(log_prob, axis=0)
    return tf.math.reduce_sum(log_prob)

def log_posterior(theta, beta):
    return log_prior(theta, beta) + log_likelihood(theta, beta)

def transform(theta, beta):
    log =  tfp.bijectors.Log()
    return [log.forward(theta), log.forward(beta)]

def transform_inverse(theta, beta):
    log =  tfp.bijectors.Log()
    return [log.inverse(theta), log.inverse(beta)]
