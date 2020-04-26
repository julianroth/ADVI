import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class MixedGauss:

    def __init__(self, id_transform=True):
        self._mu = tf.constant([[2, 2], [4, 4]], dtype=tf.float64)
        self._std = tf.constant([[[.5, .25], [.25, 1.]],
                                 [[.5, -.25], [-.25, 1.]]], dtype=tf.float64)
        self.num_params = 2
        self._id_transform = id_transform

    def log_likelihood(self, data, params):
        d1 = tfd.MultivariateNormalFullCovariance(self._mu[0, :], self._std[0, :, :])
        d2 = tfd.MultivariateNormalFullCovariance(self._mu[1, :], self._std[1, :, :])
        return tf.math.log(.5 * d1.prob(params) + .5 * d2.prob(params))

    def avg_log_likelihood(self, data, params):
        return self.log_likelihood(data, params)

    def joint_log_prob(self, data, params):
        return self.log_likelihood(data, params)

    def sep_params(self, params):
        return params

    def return_initial_state(self, random=False):
        pass

    def bijector(self):
        tfb = tfp.bijectors
        if self._id_transform:
            return tfb.Identity()
        else:
            return tfb.Shift(tf.constant(-3, dtype=tf.float64))
