
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ADVIModel(tf.keras.Model):
    """An ADVIModel object maintains all relevant information for
    performing ADVI."""

    def __init__(self, dim, log_prob, bijector, name=None):
        super(ADVIModel, self).__init__(name=name)
        # dimension of theta, mu, omega, ...
        self._dim = dim
        # target model joint log_prob
        self._log_prob = log_prob
        # function T (a tensorflow bijector)
        self._T = bijector
        # mc integration parameter, not yet supported
        self._M = 1
        # underlying Gaussian
        self._dist = tfp.distributions.Normal(np.zeros(self._dim),
                                              np.ones(self._dim))
        # model parameters
        self.mu = tf.Variable(np.zeros(self._dim), dtype=tf.float64)
        self.omega = tf.Variable(np.zeros(self._dim), dtype=tf.float64)

    def sample_eta(self, nsamples=1):
        """Produces nsamples samples from the underlying Gaussian.
        nsamples not yet supported"""
        return self._dist.sample()

    def zeta(self, eta):
        """Transforms eta samples into a zeta values."""
        return tf.add(self.mu, tf.multiply(tf.exp(self.omega), eta))

    def theta(self, zeta):
        """Transforms zeta values into a theta values."""
        return self._T.inverse(zeta)

    def sample(self):
        """Samples theta values according to current parameters."""
        return self.theta(self.zeta(self.sample_eta()))

    def gradients(self):
        """Computes the gradients for mu and omega according to the
        computations in the paper.
        Yet only supported for M=1"""
        eta = self.sample_eta()
        zeta = self.zeta(eta)
        theta = self.theta(zeta)

        # compute gradient parts of mu/omega gradients using autodiff
        # see equation (5) and (6) from left to right
        with tf.GradientTape() as t0:
            t0.watch(theta)
            res0 = self._log_prob(theta)
            grad0 = t0.gradient(res0, theta)

        with tf.GradientTape() as t1:
            t1.watch(zeta)
            res1 = self._T.inverse(zeta)
            grad1 = t1.gradient(res1, zeta)

        with tf.GradientTape() as t2:
            t2.watch(zeta)
            res2 = self._T.inverse_log_det_jacobian(zeta, 0)
            grad2 = t2.gradient(res2, zeta)

        # compute gradients according to paper
        grad_mu = tf.add(tf.multiply(grad0, grad1), grad2)
        grad_omega = tf.add(tf.multiply(grad_mu,
                            tf.multiply(eta, tf.exp(self.omega))),
                            tf.ones(self._dim, dtype=tf.float64))

        return grad_mu, grad_omega
