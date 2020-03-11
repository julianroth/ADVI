"""
This is an example implementation of HMC sampling and NUTS sampling
composed from:
- https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/HamiltonianMonteCarlo
- https://github.com/tensorflow/probability/issues/728
Important information for creating a nuts kernel:
- https://github.com/tensorflow/probability/issues/549
"""

import tensorflow as tf
import tensorflow_probability as tfp

# initialize some target distribution log probability
# just believe that this is somewhat sensible
def unnormalized_log_prob(x):
  return -x - x**2.

# set some sampling parameters
num_results = 20
num_burnin_steps = 10
init_step_size = 1.
init_state = tf.constant(1., dtype=tf.float64)

# initialize kernels
nuts_kernel = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=unnormalized_log_prob,
    step_size=init_step_size
)

hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=unnormalized_log_prob,
    num_leapfrog_steps=3,
    step_size=init_step_size
)

# initialize step size adapting kernels
# for nuts, see https://github.com/tensorflow/probability/issues/549
nuts_adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=nuts_kernel,
    num_adaptation_steps=int(num_burnin_steps * 0.8),
    step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
    step_size_getter_fn=lambda pkr: pkr.step_size,
    log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
)

hmc_adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=hmc_kernel,
    num_adaptation_steps=int(num_burnin_steps * 0.8)
)

print("Start HMC sampling...")

chain_output_mcmc = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=init_state,
    kernel=hmc_adaptive_kernel,
    trace_fn=None
)

print(chain_output_mcmc)

print("\n\nStart NUTS sampling...")

chain_output_nuts = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=init_state,
    kernel=nuts_adaptive_kernel,
    trace_fn=None
)

print(chain_output_nuts)
