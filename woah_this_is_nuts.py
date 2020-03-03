import tensorflow as tf
import tensorflow_probability as tfp
#from tensorflow_probability.mcmc import NoUTurnSampler
from models import constrained_gamma_poisson as cgp

print(cgp.data.shape)

theta = cgp.theta_prior_distribution().sample(tf.constant(cgp.U * cgp.K, dtype=tf.int64))
#theta = tf.reshape(theta, [cgp.U, cgp.K])
beta = cgp.beta_prior_distribution().sample(tf.constant(cgp.K * cgp.I, dtype=tf.int64))
#beta = tf.reshape(beta, [cgp.K, cgp.I])

#theta = tf.constant([0, 1, 2, 3, 4, 5], dtype=tf.float64, shape=(3, 2))
#beta = tf.constant([0, 1, 2, 3, 4, 5], dtype=tf.float64, shape=(2, 3))
print(theta.shape)
p = cgp.log_prior(theta, beta)
print(p)

p = cgp.log_posterior(theta, beta)

step_size = tf.concat([tf.reshape(cgp.stddev_theta(), [-1]), tf.reshape(cgp.stddev_beta(), [-1])], 0)

nuts = tfp.mcmc.NoUTurnSampler(cgp.log_posterior_nuts, step_size, parallel_iterations=1)

current_state = tf.concat([theta, beta], 0)
prev_kernel = nuts.bootstrap_results(current_state)
for i in range(10):
    print('iteration ' + (i+1))
    current_state, prev_kernel = nuts.one_step(current_state, prev_kernel)

print(cgp.log_posterior_nuts(current_state))
