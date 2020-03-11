"""
These are some unstructured HMC and NUTS sampling attempts using models
specified in Section 3.2 of our paper.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from models import dirichlet_exponential as dem
from data import frey_face

from models import constrained_gamma_poisson as cgp

# prepare unnormalised log probability function
data = frey_face.load_data()
#target_log_prob = lambda *args: dem.log_posterior_sampling(data, *args)
target_log_prob = cgp.log_posterior_nuts

#init_state = dem.initial_state()
#step_size = dem.std_step_size()
#n = (dem.U + dem.I) * dem.K
theta = cgp.theta_prior_distribution().sample(tf.constant(cgp.U * cgp.K, dtype=tf.int64))
beta = cgp.beta_prior_distribution().sample(tf.constant(cgp.K * cgp.I, dtype=tf.int64))
init_state = tf.concat([theta, beta], 0)
step_size = tf.concat([tf.reshape(cgp.stddev_theta(), [-1]), tf.reshape(cgp.stddev_beta(), [-1])], 0)
n = (dem.U + dem.I) * dem.K

#print(dem.theta_prior_distribution().stddev()[0])
#print(dem.beta_prior_distribution().stddev())

#def nuts_sampling():
# set up nuts sampler
num_results = 100
num_burnin_steps = 10
nuts = tfp.mcmc.NoUTurnSampler(target_log_prob, step_size)
state = init_state

samples = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=init_state,
    kernel=nuts,
    trace_fn=None)

print(samples[0] - samples[99])

    #kernel_results = nuts.bootstrap_results(init_state)
    #steps = 10

    # results
    #samples = np.empty((num_results, n), dtype=np.float64)

    # perform sampling
    #for t in range(num_results):
    #    state, kernel_results = nuts.one_step(state, kernel_results)
    #    samples[t] = state

    # this gives 0.0 -> sampler ain't doin' shit yet
    #print(np.sum(init_state - samples[num_results-1]))


def hmc_sampling():
    # From https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo

    # Initialize the HMC transition kernel.
    num_results = 100
    num_burnin_steps = 10
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            num_leapfrog_steps=3,
            step_size=step_size),
        num_adaptation_steps=int(num_burnin_steps * 0.8))

    # Run the chain (with burn-in).
    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=init_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    #print(samples[:,0:10])
    print(is_accepted)
