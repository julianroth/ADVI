import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# performs mc integration to estimate the predictive posterior
def predictive_posterior(model, data, theta_samples):
    log_likelihood = lambda *args: model.log_likelihood(data, *args)
    return tf.math.reduce_mean(tf.map_fn(log_likelihood, theta_samples))

def normal_samples(loc, scale, n_samples=1):
    return tfd.Normal(loc, scale).sample(n_samples)
