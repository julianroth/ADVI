import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions


class Ard:
    """Hierarchical linear regression with automatic relevance
    determination model (see Section 3.1 of ADVI paper)"""
    def __init__(self, num_features=250, transform=False):
        # number of features in data
        self.features = num_features
        # total number of trainable parameters in model
        self.num_params = (self.features * 2) + 1
        # parameters from the paper
        self.a_0 = tf.constant(1, dtype=tf.float64)
        self.b_0 = tf.constant(1, dtype=tf.float64)
        self.c_0 = tf.constant(1, dtype=tf.float64)
        self.d_0 = tf.constant(1, dtype=tf.float64)
        # set up the bijector if model is transformed into unconstrained space
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
        returns: log likelihood P(D|theta)
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
        returns: average log likelihood P(D|theta) / #data
        """
        y, _ = data
        _, ndata = y.shape
        return self.log_likelihood(data, params) / float(ndata)

    def joint_log_prob(self, data, params):
        """
        input data: (y, x) with y as target data and x as data
        params: all the parameters we are training in the model
        returns: joint log probability
        log P(theta, D) = log P(theta) + log P(D|theta)
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

    def sep_params(self, params):
        """
        input params: trained parameters for model
        returns: parameters separated in to their different types
        """
        w = params[:self.features]
        tau = params[self.features]
        alpha = params[self.features+1:]
        return w, tau, alpha

    def return_initial_state(self, random=False):
        """
        random: if true, state is initialised randomly by sampling from prior,
            if false, by taking the means of the priors
        returns: initial state in constrained or unconstrained space
        """
        if self._biji is not None:
            if random == False:
                return self._biji.forward(self._initial_state_mean())
            else:
                return self._biji.forward(self._initial_state_random())
        else:
            if random == False:
                return self._initial_state_mean()
            else:
                return self._initial_state_random()

    def _initial_state_random(self):
        """
        computes an initial parameter state by randomly sampling from the priors
        """
        alpha = self.alpha_prior_().sample([self.features])
        tau = self.tau_prior_().sample()
        sigma = tf.math.sqrt(tau)
        tau = tf.reshape(tau, [1])
        one_over_sqrt_alpha = self.convert_alpha(alpha)
        w = self.w_prior_(sigma, one_over_sqrt_alpha).sample()
        return tf.concat([w, tau, alpha], 0)

    def _initial_state_mean(self):
        """
        computes an initial parameter state by taking the means of the priors
        """
        alpha = self.alpha_prior_().mean() * np.ones([self.features])
        tau = tf.constant(1., dtype=tf.float64)
        sigma = tf.math.sqrt(tau)
        tau = tf.reshape(tau, [1])
        one_over_sqrt_alpha = self.convert_alpha(alpha)
        w = self.w_prior_(sigma, one_over_sqrt_alpha).mean()
        return tf.concat([w, tau, alpha],0)
        
    def bijector(self):
        """
        transformation function associated with this model
        """
        return tfp.bijectors.Blockwise([tfp.bijectors.Identity(), tfp.bijectors.Log()],
                                        [self.features, self.features+1])
