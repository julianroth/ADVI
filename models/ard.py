import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

# Template for models, things I need
# 1. self.features = num_features
# number of features in the data
# i.e. x has 10 features
#
# 2. self.num_params
# number of trainable parameters in model. 
# 3. funtions all functions below log_likelihood
# 
#


class Ard:
    def __init__(self, num_features=250, transform=False):
        # number of features in data
        self.features = num_features
        # total number of trainable parameters in model
        self.num_params = (self.features *2) + 1
        self.a_0 = tf.constant(1, dtype=tf.float64)
        self.b_0 = tf.constant(1, dtype=tf.float64)
        self.c_0 = tf.constant(1, dtype=tf.float64)
        self.d_0 = tf.constant(1, dtype=tf.float64)
        self._biji = self.bijector() if transform else None
        
    def convert_alpha(self, alpha):
        one_over_sqrt_alpha  = tf.map_fn(
            lambda x:  1/(tf.math.sqrt(x)), alpha, dtype=tf.float64)
        return one_over_sqrt_alpha
    
    def alpha_prior_(self):
        return tfd.Gamma(self.a_0, self.b_0)

    def tau_prior_(self):
        return tfd.InverseGamma(self.c_0, self.d_0)
        
    def w_prior_(self, sigma, one_over_sqrt_alpha):
        return tfd.Normal(0,tf.math.multiply(sigma, one_over_sqrt_alpha))

    def log_likelihood(self, data, params):
        """
        input data: (y, x) with y as target data and x as data
        input params: all trainable parameters in model
        returns: log likelihood
        P(D|theta)
        """
        if self._biji is not None:
            params = self._biji.inverse(params)
        y, x = data
        w, tau, alpha = self.sep_params(params)
        alpha_prior = self.alpha_prior_()
        one_over_sqrt_alpha = self.convert_alpha(alpha)
        tau_prior = self.tau_prior_()
        sigma = tf.math.sqrt(tau)
        w_prior = self.w_prior_(sigma, one_over_sqrt_alpha)
        log_likelihood_ = tf.reduce_sum(tfd.Normal(
            tf.linalg.matvec(x, w, transpose_a=True), sigma).log_prob(y))
        return log_likelihood_

    def avg_log_likelihood(self, data, params):
        """
        input data: (y, x) with y as target data and x as data
        input params: all trainable parameters in model
        returns: average log likelihood
        P(D|theta) / #data
        """
        y, _ = data
        _, ndata = y.shape
        return self.log_likelihood(data, params) / float(ndata)

    def joint_log_prob(self, data, params):
        """
        input data: (y, x) with y as target data and x as data
        params: all the parameters we are training in the model
        returns: joint log probability
        joint log prob.
        log P(theta, D) = log P(theta) + log P(D|theta)
        params is a tensor of size (features * 2) + 1
        regressors / weights  = [:features]
        tau = [features+1]
        alpha = [features+1:]
        """
        y, x = data
        if self._biji is not None:
            params = self._biji.inverse(params)
        w, tau, alpha = self.sep_params(params)
        alpha_prior = self.alpha_prior_()
        tau_prior = self.tau_prior_()
        sigma = tf.math.sqrt(tau)
        one_over_sqrt_alpha = self.convert_alpha(alpha)
        w_prior = self.w_prior_(sigma, one_over_sqrt_alpha)
        log_likelihood_ = tf.reduce_sum(tfd.Normal(
            tf.linalg.matvec(x, w, transpose_a=True), sigma).log_prob(y))
        sum_is = log_likelihood_ + tf.reduce_sum(w_prior.log_prob(w)) + tau_prior.log_prob(tau) + tf.reduce_sum(alpha_prior.log_prob(alpha)) 
        return sum_is
        
    def loss(self, y, x, w):
        """
        input y: data target
        input x: data
        input w: trained parameters
        returns: squared difference between target and predicted
        good for debugging
        just does sum((y - ypred)**2)
        """
        y_pred = tf.linalg.matvec(x, w, transpose_a=True)
        return tf.reduce_sum(tf.math.abs(tf.math.subtract(y, y_pred)))

    def sep_params(self, params):
        """
        input params: trained parameters for model
        returns: parameters separated in to their different types
        """
        w = params[:self.features]
        tau = params[self.features+1]
        alpha = params[self.features+1:]
        return w, tau, alpha

    def return_initial_state(self):
        """
        Returns: starting states for HMC and Nuts by sampling from prior
        distribution
        """
        if self._biji is not None:
            return self._biji.forward(self._initial_state_mean())
        else:
            return self._initial_state_mean()
        

    def _initial_state_random(self):
        alpha = self.alpha_prior_().sample([self.features])
        tau = self.tau_prior_().sample()
        sigma = tf.math.sqrt(tau)
        tau = tf.reshape(tau, [1])
        one_over_sqrt_alpha = self.convert_alpha(alpha)
        w = self.w_prior_(sigma, one_over_sqrt_alpha).sample()
        return tf.concat([w, tau, alpha], 0)

    def _initial_state_mean(self):
        alpha = self.alpha_prior_().mean() * np.ones([self.features])
        tau = tf.constant(1., dtype=tf.float64)
        sigma = tf.math.sqrt(tau)
        tau = tf.reshape(tau, [1])
        one_over_sqrt_alpha = self.convert_alpha(alpha)
        w = self.w_prior_(sigma, one_over_sqrt_alpha).mean()
        return tf.concat([w, tau, alpha],0)

    def bijector(self):
        """
        returns: bijector associated with this model
        """
        return tfp.bijectors.Blockwise([tfp.bijectors.Identity(), tfp.bijectors.Log()],
                                        [self.features, self.features+1])
        
