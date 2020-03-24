import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions


class DirichletExponential:

    def __init__(self, users=28, items=20, factors=10):
        # dimensions from paper (see Section 3.2):
        # U = 28, I = 20, K = 10
        self._U = users
        self._I = items
        self._K = factors
        # number of features in data
        self.features = self._U * self._I
        # total number of trainable parameters in model
        self.num_params = (self._U + self._I) * self._K
        # parameter settings from paper
        self._alpha_0 = tf.constant(1000. * np.ones(self._K), dtype=tf.float64)
        self._lambda_0 = tf.constant(0.1, dtype=tf.float64)

    def theta_prior(self):
        return tfd.Dirichlet(self._alpha_0)

    def beta_prior(self):
        return tfd.Exponential(self._lambda_0)

    def likelihood(self, rate):
        return tfd.Poisson(rate)

    def std_step_sizes(self):
        """
        returns: A vector of standard deviations from the priors of
        the parameters. The vector has the same shape as the parameter
        vector.
        """
        std_theta = self.theta_prior().stddev()[0]
        std_beta = self.beta_prior().stddev()
        step_size = tf.concat([std_theta * np.ones(self._U*self._K), std_beta * np.ones(self._I*self._K)], 0)
        return step_size

    def log_likelihood(self, data, params):
        """
        returns: log likelihood P(D | theta, beta)
        """
        theta, beta = self.sep_params(params)
        theta = tf.reshape(theta, [self._U, self._K])
        beta = tf.reshape(beta, [self._K, self._I])
        rates = tf.reshape(tf.linalg.matmul(theta, beta), [-1])
        return tf.reduce_sum(self.likelihood(rates).log_prob(data))

    def joint_log_prob(self, data, params):
        """
        returns: joint log probability
        log P(data, theta, beta) = log P(D | theta, beta)
                                   + log P(theta) + log P(beta)
        """
        log_like = self.log_likelihood(data, params)
        theta, beta = self.sep_params(params)
        theta = tf.reshape(theta, [self._U, self._K])
        log_theta_prior = tf.reduce_sum(self.theta_prior().log_prob(theta))
        log_beta_prior = tf.reduce_sum(self.beta_prior().log_prob(beta))
        return log_like + log_theta_prior + log_beta_prior

    def sep_params(self, params):
        """
        input params: trained parameters for model
        returns: parameters separated in to their different types
        """
        theta = params[:self._U * self._K]
        beta = params[self._U * self._K:]
        return theta, beta

    def return_initial_state(self):
        """
        returns: starting states for sampling/optimisation from prior
        distributions
        """
        theta = self.theta_prior().sample(self._U)
        beta = self.beta_prior().sample(self._I * self._K)
        return tf.concat([tf.reshape(theta, [-1]), beta], 0)

    def bijector(self):
        """
        returns: bijector associated with this model
        """
        return tfp.bijectors.Log()
