import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

class Ard:
    def __init__(self, y_train, x_train, y_test, x_test):
        self.y = y_train
        self.x = x_train
        self.y_test = y_test
        self.x_test = x_test
        self.features = self.x.shape[0]
        self.a_0 = tf.constant(1, dtype=tf.float64)
        self.b_0 = tf.constant(1, dtype=tf.float64)
        self.c_0 = tf.constant(1, dtype=tf.float64)
        self.d_0 = tf.constant(1, dtype=tf.float64)
        
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
        

    def joint_log_prob(self, params):
        
        # Must check
        # input current estimates of parameters, data,
        # return log joint distribution
        w, tau, alpha = sep_params(params, self.features)
        alpha_prior = self.alpha_prior_()
        one_over_sqrt_alpha = self.convert_alpha(alpha)
        tau_prior = self.tau_prior_()
        sigma = tf.math.sqrt(tau)
        w_prior = self.w_prior_(sigma, one_over_sqrt_alpha)
        log_likelihood = tfd.Normal(
            tf.linalg.matvec(self.x, w, transpose_a=True), sigma).log_prob(self.y)
        sum_alpha_prior = tf.reduce_sum(alpha)
        sum_is = tf.reduce_sum(
            log_likelihood) + tf.reduce_sum(w_prior.log_prob(w)) + tau_prior.log_prob(tau) + tf.reduce_sum(alpha_prior.log_prob(alpha)) 
        return sum_is
        
    def some_kind_of_loss(self, w):
        y_pred = tf.linalg.matvec(self.x_test, w, transpose_a=True)
        return tf.reduce_sum(tf.math.abs(tf.math.subtract(self.y_test, y_pred)))
        
    def log_likelihood(self, params):
        w, tau, alpha = sep_params(params, self.features)
        alpha_prior = self.alpha_prior_()
        one_over_sqrt_alpha = self.convert_alpha(alpha)
        tau_prior = self.tau_prior_()
        sigma = tf.math.sqrt(tau)
        w_prior = self.w_prior_(sigma, one_over_sqrt_alpha)
        log_likelihood_ = tf.reduce_sum(tfd.Normal(
            tf.linalg.matvec(self.x_test, w, transpose_a=True), sigma).log_prob(self.y_test))
        return log_likelihood_

    def return_initial_state(self):
        alpha = self.alpha_prior_().sample([self.features])
        tau = self.tau_prior_().sample()
        sigma = tf.math.sqrt(tau)
        tau = tf.reshape(tau, [1])
        one_over_sqrt_alpha = self.convert_alpha(alpha)
        w = self.w_prior_(sigma, one_over_sqrt_alpha).sample()
        return tf.concat([w, tau, alpha],0)
     

def sep_params(matrix, features):
    w = matrix[:features]
    tau = matrix[features+1]
    alpha = matrix[features+1:]
    return w, tau, alpha
