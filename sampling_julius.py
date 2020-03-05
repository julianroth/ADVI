import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from models import dirichlet_exponential as dem
from data import frey_face

# prepare unnormalised log probability function
data = frey_face.load_data()
target_log_prob = lambda *args: dem.log_posterior_sampling(data, *args)

init_state = dem.initial_state()
n = (dem.U + dem.I) * dem.K


def nuts_sampling():
    # set up nuts sampler
    step_size = tf.ones(n)
    nuts = tfp.mcmc.NoUTurnSampler(target_log_prob, step_size)
    state = init_state
    kernel_results = nuts.bootstrap_results(init_state)
    steps = 10

    # results
    samples = np.empty((steps, n), dtype=np.float64)

    # perform sampling
    for t in range(steps):
        state, kernel_results = nuts.one_step(state, kernel_results)
        samples[t] = state

    # this gives 0.0 -> sampler ain't doin' shit yet
    print(np.sum(init_state - samples[steps-1]))


def hmc_sampling():
    # From https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo

    # Initialize the HMC transition kernel.
    num_results = 100
    num_burnin_steps = 10
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            num_leapfrog_steps=3,
            step_size=1.),
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
