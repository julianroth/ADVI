import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ADVIModel(tf.keras.Model):
    """An ADVIModel object maintains all relevant information for
    performing ADVI."""

    def __init__(self, dim, log_prob, bijector, m=1, name=None):
        super(ADVIModel, self).__init__(name=name)
        # dimension of theta, mu, omega, ...
        self._dim = dim
        # target model joint log_prob
        self._log_prob = log_prob
        # function T (a tensorflow bijector)
        self._T = bijector
        self._inv_log_det_T = lambda x: self._T.inverse_log_det_jacobian(x, 1)
        # number of samples for approximating expectation in gradients
        self._M = m
        # underlying Gaussian
        self._dist = tfp.distributions.Normal(np.zeros(self._dim),
                                              np.ones(self._dim))
        # model parameters
        self.mu = tf.Variable(np.zeros(self._dim), dtype=tf.float64, trainable=True)
        self.omega = tf.Variable(np.zeros(self._dim), dtype=tf.float64, trainable=True)

    def _sample_eta(self, nsamples=1):
        """Produces samples from the underlying Normal distribution.
            :arg nsamples: (int) Number of samples (1 by default).
            :return A sample matrix of shape (nsamples, self._dim).
        """
        return self._dist.sample(nsamples)

    def _zeta(self, eta):
        """Transforms eta samples into a zeta samples.
            :arg eta: A sample matrix of shape (nsamples, self._dim).
            :return A sample matrix of shape (nsamples, self._dim).
        """
        return tf.add(self.mu, tf.multiply(tf.exp(self.omega), eta))

    def _theta(self, zeta):
        """Transforms zeta samples into a theta samples.
            :arg zeta: A sample matrix of shape (nsamples, self._dim).
            :return A sample matrix of shape (nsamples, self._dim).
        """
        return self._T.inverse(zeta)

    def sample(self, nsamples=1):
        """Samples theta values according to the current parameters.
            :arg nsamples: (int) Number of samples (1 by default).
            :return A sample matrix of shape (nsamples, self._dim).
        """
        return self._theta(self._zeta(self._sample_eta(nsamples)))

    def elbo(self, nsamples=-1):
        """Approximates the elbo function for the current mu and omega
        values according to the computations in the paper (Eq. (4)).
        MC integration is used for computing the expectations.
            :arg nsamples: (int) Number of eta samples to approximate
                the expectation in the elbo. Per default, the attribute
                self._M is used.
            :return The one-dimensional approximate value of the elbo
                function.
        """
        if nsamples == -1:
            nsamples = self._M
        eta = self._sample_eta(nsamples)
        zeta = self._zeta(eta)
        theta = self._theta(zeta)

        inner = tf.add(tf.map_fn(self._log_prob, theta),
                       tf.map_fn(self._inv_log_det_T, zeta))
        assert inner.shape == (nsamples,)

        return tf.reduce_mean(inner) + tf.reduce_sum(self.omega)

    def neg_elbo(self, nsamples=-1):
        elbo = self.elbo(nsamples=nsamples)
        return -elbo

    def gradients(self, nsamples=-1):
        """Approximates the gradients for mu and omega according
        to the computations in the paper (Eq. (5), (6)). MC integration
        is used for computing the expectations.
            :arg nsamples: (int) Number of eta samples to approximate
                the expectation in the gradients (corresponds to M in
                the paper). Per default, the attribute self._M is used.
            :return (grad_mu, grad_omega), where both are TensorFlow
                vectors of with self._dim dimensions.
        """
        if nsamples == -1:
            nsamples = self._M
        eta = self._sample_eta(nsamples)
        zeta = self._zeta(eta)
        theta = self._theta(zeta)

        # compute gradient parts of mu/omega gradients using autodiff
        # see equation (5) and (6) from left to right
        with tf.GradientTape() as t0:
            t0.watch(theta)
            res0 = tf.map_fn(self._log_prob, theta)
            grad0 = t0.gradient(res0, theta)
            assert res0.shape == (nsamples,)
            assert grad0.shape == (nsamples, self._dim)
            
        with tf.GradientTape() as t1:
            t1.watch(zeta)
            res1 = self._T.inverse(zeta)
            grad1 = t1.gradient(res1, zeta)
            # maybe Jacobian needed instead.. but I don't think so
            assert res1.shape == (nsamples, self._dim)
            assert grad1.shape == (nsamples, self._dim)
            
        with tf.GradientTape() as t2:
            t2.watch(zeta)
            res2 = tf.map_fn(self._inv_log_det_T, zeta)
            grad2 = t2.gradient(res2, zeta)
            assert res0.shape == (nsamples,)
            assert grad0.shape == (nsamples, self._dim)

        # compute gradients according to paper
        inner = tf.add(tf.multiply(grad0, grad1), grad2)
        outer = tf.multiply(inner, tf.multiply(eta, tf.exp(self.omega)))
        grad_mu = tf.reduce_mean(inner, 0)
        grad_omega = tf.add(tf.reduce_mean(outer, 0), 1.)
        assert grad_mu.shape == (self._dim,)
        assert grad_omega.shape == (self._dim,)

        return grad_mu, grad_omega
