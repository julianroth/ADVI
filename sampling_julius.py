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

step_size = tf.ones(n)

# set up nuts sampler
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
