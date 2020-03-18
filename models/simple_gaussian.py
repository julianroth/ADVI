
import tensorflow as tf
import tensorflow_probability as tfp


class SimpleGaussian:

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dim = self.mu.shape
        self.dist = tfp.distributions.Normal(self.mu, self.sigma)

    def log_prob(self, params):
        return tf.reduce_sum(self.dist.log_prob(params))

    def bijector(self):
        return tfp.bijectors.Identity()
