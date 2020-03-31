import tensorflow as tf
import tensorflow_probability as tfp
from data import frey_face
from utils.bijectors import LogOrdered, positive_ordered
from utils.sep_data import sep_training_test
tfd = tfp.distributions

# This implements the constrained gamma poisson distribution from 3.2
class Gamma_Poisson():
    def __init__(self, num_test=-1, test_split=0.2, permute=False):
        self._data = frey_face.load_data()

        self._train_data, self._test_data =\
            sep_training_test(self._data, num_test=num_test, test_split=test_split, permute=permute)

        self._K = 10
        self._U = 28
        self._I = 20
        self.num_params = self._U*self._K + self._K*self._I

        # Gamma priors
        self._a_0 = tf.constant(1, dtype=tf.float64)
        self._b_0 = tf.constant(1, dtype=tf.float64)
        self._c_0 = tf.constant(1, dtype=tf.float64)
        self._d_0 = tf.constant(1, dtype=tf.float64)

        self.theta_prior = tfd.Gamma(self._a_0, self._b_0)
        self.beta_prior = tfd.Gamma(self._c_0, self._d_0)

    def prior_log_prob(self, params):
        theta, beta = self.sep_params(params)
        theta_prob = self.theta_prior.log_prob(theta)
        beta_prob = self.beta_prior.log_prob(beta)
        return tf.math.reduce_sum(theta_prob) + tf.math.reduce_sum(beta_prob)

    def log_likelihood(self, data, params):
        theta, beta = self.sep_params(params)
        theta = tf.reshape(theta, [self._U, self._K])
        beta = tf.reshape(beta, [self._K, self._I])
        poisson_params = tf.linalg.matmul(theta, beta)
        poisson_params = tf.reshape(poisson_params, [-1])
        likelihood_distr = tfd.Poisson(poisson_params)
        log_prob = likelihood_distr.log_prob(data)
        #log_prob = tf.math.reduce_mean(log_prob, axis=0)
        return tf.math.reduce_sum(log_prob)

    def avg_log_likelihood(self, data, params):
        ndata, _ = data.shape
        return self.log_likelihood(data, params) / float(ndata * self._U * self._I)

    def joint_log_prob(self, data, params):
        return self.prior_log_prob(params) + self.log_likelihood(data, params)

    def sep_params(self, params):
        theta = params[:self._U*self._K]
        beta = params[self._U*self._K:]
        return theta, beta

    def concat_params(self, theta, beta):
        theta = tf.reshape(theta, [self._U * self._K])
        beta = tf.reshape(beta, [self._K * self._I])
        return tf.concat([theta, beta], axis=0)

    def return_initial_state(self, random=False):
        if random:
            return tf.concat([sorted(self.theta_prior.sample(self._K)) for _ in range(self._U)] +
                             [self.beta_prior.sample(self._K * self._I)], axis=0)
        else:
            theta_init = tf.constant([0.2 + i * 0.2 for i in range(self._K)], dtype=tf.float64)
            return tf.concat([tf.tile(theta_init, [self._U]),
                              tf.repeat(self.beta_prior.mean(), self._K * self._I)], axis=0)

    def bijector(self, ordered=True):
        tfb = tfp.bijectors
        if ordered:
            #simpl = LogOrdered()
            simpl = positive_ordered()
            res_orig = tfb.Reshape([self._U, self._K])
            res_trans = tfb.Invert(tfb.Reshape([self._U, self._K]))
            ordered_log = tfb.Chain([res_trans, simpl, res_orig])
            return tfb.Blockwise([ordered_log, tfb.Log()], [self._U * self._K, self._K * self._I])
        else:
            return tfb.Log()
