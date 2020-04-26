import unittest
import tensorflow as tf
from advi import core
from models import simple_gaussian


class GaussianPosterior(unittest.TestCase):
    """This test tries to find artificial Gaussian posterior distributions
    via ADVI.
    """

    def one_dim(self):
        mu = tf.constant(4., dtype=tf.float64)
        sigma = tf.constant(0.5, dtype=tf.float64)

        target = simple_gaussian.SimpleGaussian(mu, sigma)

        res = core.run_advi(1, target.log_prob, target.bijector(), epsilon=0.0001)

        print("Target:\n  mu:     {}\n  sigma: {}".format(mu.numpy(), sigma.numpy()))
        print("ADVI result:\n  mu:     {}\n  sigma: {}".format(res.mu.numpy(), tf.exp(res.omega).numpy()))

    def mult_dim(self):
        dim = 5
        mu = tf.random.uniform((dim,), minval=-10., maxval=10., dtype=tf.dtypes.float64)
        sigma = tf.random.uniform((dim,), minval=0., maxval=20., dtype=tf.dtypes.float64)

        target = simple_gaussian.SimpleGaussian(mu, sigma)

        res = core.run_advi(dim, target.log_prob, target.bijector(), epsilon=0.0001, step_limit=10000)

        print("Target:\n  mu:     {}\n  sigma: {}".format(mu.numpy(), sigma.numpy()))
        print("ADVI result:\n  mu:     {}\n  sigma: {}".format(res.mu.numpy(), tf.exp(res.omega).numpy()))
