import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import data.election88 as polls
from utils.sep_data import sep_training_test
tfd = tfp.distributions

class HLR:
    def __init__(self, num_test=-1, test_split=0.2, permute=False):
        self._x, self._y = polls.load_data()
        self._prev_vote = polls.load_prev_vote()

        self._train_data, self._test_data =\
            sep_training_test(self._x, self._y, num_test=num_test, test_split=test_split, permute=permute)

        # lower and upper bound for stds
        self._ulb = tf.constant(0, dtype=tf.float64)
        self._uub = tf.constant(100, dtype=tf.float64)

        # mean and std for beta parameters
        self._beta_mean = tf.constant(0, dtype=tf.float64)
        self._beta_std = tf.constant(100, dtype=tf.float64)

        # number of respective categories
        self._n_age = 4
        self._n_edu = 4
        self._n_age_edu = self._n_age * self._n_edu
        self._n_region = 5
        self._n_state = 51

        # number of parameters
        self._n_alpha = self._n_age + self._n_edu + self._n_age_edu + self._n_region + self._n_state
        self._n_beta = 5
        self.num_params = 2*self._n_alpha + self._n_beta

        # Copied from the book's code
        # Regions:  1=northeast, 2=south, 3=north central, 4=west, 5=d.c.
        # We have to insert d.c. (it is the 9th "state" in alphabetical order)
        self._regions = np.array([3,4,4,3,4,4,1,1,5,3,3,4,4,2,2,2,2,3,3,1,1,1,2,2,3,2,4,2,4,1,1,4,1,3,2,2,3,4,1,1,3,2,3,3,4,1,3,4,1,2,4],\
            dtype=int) - 1

    def std_prior(self):
        return tfd.Uniform(self._ulb, self._uub)

    def beta_prior(self):
        return tfd.Normal(self._beta_mean, self._beta_std)

    # prior for all alpha priors but alpha state
    def alpha_prior(self, stds):
        std = stds[:-self._n_state]
        return tfd.Normal(0, std)

    def alpha_state_prior(self, betas, alphas, stds):
        beta_prev_vote = betas[-1]
        s = self._n_age + self._n_edu + self._n_age_edu
        alpha_region = alphas[s:s + self._n_region]
        std_state = stds[-self._n_state:]
        return tfd.Normal(tf.gather(alpha_region, self._regions) + beta_prev_vote * self._prev_vote, std_state)


    def log_prior(self, params):
        betas, alphas, stds = self.sep_params(params)
        alpha_state = alphas[-self._n_state:]
        alphas_no_state = alphas[:-self._n_state]
        beta_log_prob = tf.math.reduce_sum(self.beta_prior().log_prob(betas))
        alpha_log_prob = tf.math.reduce_sum(self.alpha_prior(stds).log_prob(alphas_no_state)) +\
                         tf.math.reduce_sum(self.alpha_state_prior(betas, alphas, stds).log_prob(alpha_state))
        std_log_prob = tf.math.reduce_sum(self.std_prior().log_prob(stds))
        return beta_log_prob + alpha_log_prob + std_log_prob

    def log_likelihood(self, data, params):
        x, y = data
        betas, alphas, stds = self.sep_params(params)
        alpha_age, alpha_edu, alpha_age_edu, alpha_region, alpha_state = self.sep_alphas(alphas)
        regions = self._regions[x[:, 0]]
        age_edu = self._n_age * x[:, 2] + x[:, 1]
        y_hat = betas[0] + betas[1] * x[:, 3] + betas[2] * x[:, 4] + tf.gather(alpha_region, regions) + tf.gather(alpha_age, x[:, 2]) +\
                tf.gather(alpha_edu, x[:, 1]) + tf.gather(alpha_age_edu, age_edu) + tf.gather(alpha_state, x[:, 0])
        return tf.math.reduce_sum(tf.math.log_sigmoid(y_hat))

    def avg_log_likelihood(self, data, params):
        # TODO implement
        return self.log_likelihood(data, params)

    def joint_log_prob(self, data, params):
        return self.log_prior(params) + self.log_likelihood(data, params)

    def sep_params(self, params):
        """
        input params: trained parameters for model
        returns: parameters separated in to their different types
        """
        betas = params[0:self._n_beta]
        alphas = params[self._n_beta:self._n_beta + self._n_alpha]
        stds = params[self._n_beta + self._n_alpha:]
        return betas, alphas, stds

    def concat_params(self, betas, alphas, stds):
        return tf.concat([betas, alphas, stds], axis=0)

    def sep_alphas(self, alphas):
        from_index = 0
        alpha_age = alphas[from_index:from_index + self._n_age]
        from_index += self._n_age
        alpha_edu = alphas[from_index:from_index + self._n_edu]
        from_index += self._n_edu
        alpha_age_edu = alphas[from_index:from_index + self._n_age_edu]
        from_index += self._n_age_edu
        alpha_region = alphas[from_index:from_index + self._n_region]
        from_index += self._n_region
        alpha_state = alphas[from_index:from_index + self._n_state]
        return alpha_age, alpha_edu, alpha_age_edu, alpha_region, alpha_state

    def return_initial_state(self, random=False):
        """
        Returns: starting states for HMC and Nuts by sampling from prior
        distribution
        """
        # currently no special state for alpha_state
        # mean of stds for alpha prior
        stds = self.std_prior().mean() * tf.ones([self._n_alpha], dtype=tf.float64)
        if random:
            return tf.concat([self.beta_prior().sample(self._n_beta),
                              self.alpha_prior(stds).sample(self._n_alpha),
                              self.std_prior().sample(self._n_alpha)], axis=0)
        else:
            return tf.concat([self.beta_prior().mean() * tf.ones([self._n_beta], dtype=tf.float64),
                              self.alpha_prior(stds).mean() * tf.ones([self._n_alpha], dtype=tf.float64),
                              self.std_prior().mean() * tf.ones([self._n_alpha], dtype=tf.float64)], axis=0)

    def bijector(self):
        tfb = tfp.bijectors
        return tfb.Blockwise([tfb.Identity(), tfb.Invert(tfb.Sigmoid(self._ulb, self._uub))], [self._n_beta + self._n_alpha, self._n_alpha])
