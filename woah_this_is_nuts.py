import tensorflow as tf
import tensorflow_probability as tfp
#from tensorflow_probability.mcmc import NoUTurnSampler
from models import constrained_gamma_poisson as cgp
from data import frey_face

data = frey_face.load_data()

# set some sampling parameters
num_results = 20
num_burnin_steps = 10
init_step_size = cgp.transform(*cgp.step_size())
init_state = cgp.transform(*cgp.initial_state())

def p(log_prob, transform):
    return lambda s1, s2: log_prob(*transform(s1, s2))

# initialize kernels
nuts_kernel = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=p(cgp.log_posterior, cgp.transform_inverse),
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

print("\n\nStart NUTS sampling...")

def state_saver(current_state, kernel):
    return current_state

chain_output_nuts = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=init_state,
    kernel=nuts_adaptive_kernel,
    trace_fn=state_saver
)

print(chain_output_nuts)
print("TEST\n\n")
print(cgp.log_posterior(*cgp.transform_inverse(*init_state)))
for i in range(num_results):
    theta, beta = cgp.transform_inverse(chain_output_nuts[0][i, :], chain_output_nuts[1][i, :])
    print(cgp.log_posterior(theta, beta))
