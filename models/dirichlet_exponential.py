import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

# Values from paper
U = 28
I = 20
K = 10
alpha_0 = tf.constant(1000 * np.ones(K), dtype=tf.float64)
lambda_0 = tf.constant(0.1, dtype=tf.float64)


def theta_prior_distribution():
    return tfd.Dirichlet(alpha_0)


def beta_prior_distribution():
    return tfd.Exponential(lambda_0)


def likelihood_distribution(rate):
    return tfd.Poisson(rate)


# Computes the unnormalised log posterior probability.
def log_posterior(data, theta, beta):
    theta = tf.reshape(theta, [U, K])
    beta = tf.reshape(beta, [K, I])
    rates = tf.reshape(tf.linalg.matmul(theta, beta), [-1])
    post_dist = likelihood_distribution(rates)
    log_post_prob = tf.reduce_sum(post_dist.log_prob(data))
    return log_post_prob


def log_posterior_sampling(data, params):
    return log_posterior(data, params[:U*K], params[U*K:])


def initial_state():
    theta = theta_prior_distribution().sample(U)
    beta = beta_prior_distribution().sample(I*K)
    params = tf.concat([tf.reshape(theta, [-1]), beta], 0)
    return params
