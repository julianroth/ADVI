import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from models.ard import joint_log_prob, return_initial_state, sep_params
# NEXT TO DO MUST CHANGE INITAL CHAIN STATES TO SAMPLE FROM DISTRIBUTION
# Target distribution is proportional to: `exp(-x (1 + x))`.

# Initialize the HMC transition kernel.
def make_training_data(num_samples, dims, sigma):
  x = np.random.randn(dims, num_samples).astype(np.float64)
  w = sigma * np.random.randn(1, dims).astype(np.float64)
  noise = np.random.randn(num_samples).astype(np.float64)
  y = w.dot(x) + noise
  return y, x, w

def sep_training_test(y,x, test):
  y_train = y[:test]
  y_test = y[test:]
  x_train = x[:test]
  x_test = x[:test]
  return y_train, y_test, x_train, x_test


y, x, w = make_training_data(100, 10, 0.5)
print("y shape", y.shape)
print("x shape", x.shape)
print("w shape", w.shape)

y_train, y_test, x_train, x_test = sep_training_test(y,x,10)
print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)
initial_chain_state = return_initial_state(10)
joint_log_prob2 = lambda *args: joint_log_prob(y_train, x_train, *args)
num_results = int(10e3)
num_burnin_steps = int(1e3)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=joint_log_prob2,
        num_leapfrog_steps=3,
        step_size=1.),
    num_adaptation_steps=int(num_burnin_steps * 0.8))

nuts = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=joint_log_prob2, step_size=1., max_tree_depth=10, max_energy_diff=1000.0,
    unrolled_leapfrog_steps=1, parallel_iterations=10, seed=None, name=None
)
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

@tf.function
def run_chain_nuts():
  print("running chain")
  # Run the chain (with burn-in).
  states, is_accepted = tfp.mcmc.sample_chain(
      num_results=20,
      num_burnin_steps=num_burnin_steps,
      current_state=initial_chain_state,
      kernel=nuts)
  return states, is_accepted

print(" will start to run chain!!!!!!!!!!!!!!!!!!")

states, is_accepted = run_chain_nuts()

print(is_accepted)

print("states shape", states.shape)
sample_mean = tf.reduce_mean(states, axis=[0]) 
w, tau, alpha = sep_params(sample_mean, 10)
print("w shape", w.shape)
print("tau shape", tau.shape)
print("alpha shape", alpha.shape)
print("check4")
#print('mean:{:.4f}  stddev:{:.4f}'.format(
#    sample_mean.numpy(), sample_stddev.numpy()))
