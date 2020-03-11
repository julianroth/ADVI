import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from models.ard import joint_log_prob, return_initial_state
# NEXT TO DO MUST CHANGE INITAL CHAIN STATES TO SAMPLE FROM DISTRIBUTION
# Target distribution is proportional to: `exp(-x (1 + x))`.

# Initialize the HMC transition kernel.
def make_training_data(num_samples, dims, sigma):
  x = np.random.randn(dims, num_samples).astype(np.float64)
  w = sigma * np.random.randn(1, dims).astype(np.float64)
  noise = np.random.randn(num_samples).astype(np.float64)
  y = w.dot(x) + noise
  return y, x, w

y, x, w = make_training_data(100, 10, 0.5)
print("x is:", x.shape)
initial_chain_state = return_initial_state(10)

print("check0")
joint_log_prob2 = lambda *args: joint_log_prob(y, x, *args)
print("check1")
num_results = int(10e3)
num_burnin_steps = int(1e3)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=joint_log_prob2,
        num_leapfrog_steps=3,
        step_size=1.),
    num_adaptation_steps=int(num_burnin_steps * 0.8))

print("check2")
# Run the chain (with burn-in).
@tf.function
def run_chain():
  print("run chain")
  # Run the chain (with burn-in).
  states, is_accepted = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=initial_chain_state,
      kernel=adaptive_hmc,
    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)
  return states, is_accepted
#  sample_mean = tf.reduce_mean(states, axis=0)
#  sample_stddev = tf.sqrt(tf.reduce_mean(
#    tf.squared_difference(states, sample_mean),
#    axis=0))
    


print(" will start to run chain!!!!!!!!!!!!!!!!!!")

sample_mean, is_accepted = run_chain()
print(is_accepted)
print("check4")
#print('mean:{:.4f}  stddev:{:.4f}'.format(
#    sample_mean.numpy(), sample_stddev.numpy()))
