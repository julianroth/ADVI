import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from advi.model import ADVIModel
from advi.core import run_advi
from models.ard import Ard
import os

#tf.config.experimental_run_functions_eagerly(True)

# ## Making training data
def make_training_data(num_samples, dims, sigma):
  """
  Creates training data when half of the regressors are 0
  """
  x = np.random.randn(dims, num_samples).astype(np.float64)
  w = sigma * np.random.randn(1, dims).astype(np.float64)
  noise = np.random.randn(num_samples).astype(np.float64)
  w[:,:int(dims/2)] = 0
  y = w.dot(x) + noise
    
  return y, x, w

def sep_training_test(y,x,test):
  y_train = y[:,test:]
  x_train = x[:,test:]
  
  y_test = y[:,:test]
  x_test = x[:,:test]
  return y_train, y_test, x_train, x_test
what_to_run = "hmc"
num_features = 200

y, x, w = make_training_data(1000, num_features, 2)
y_train, y_test, x_train, x_test = sep_training_test(y,x,100)
step_size_hmc = 0.001

print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)
data_train = (y_train, x_train)
data_test = (y_test, x_test)

def state_to_log_like(states, data, model):
    sample_mean = tf.reduce_mean(states, axis=[0])
    return model.log_likelihood(data, states)

def state_to_loss(states, data, model):
    sample_mean = tf.reduce_mean(states, axis=[0])
    w, _, _ = sep_params(states, num_features)
    return model.loss(data, w)


# Defining summary writer
summary_writer = tf.compat.v2.summary.create_file_writer('/tmp/summary_chain', flush_millis=10000)

# trace_functions for hmc and nuts
def trace_fn(state, results):
  with tf.compat.v2.summary.record_if(tf.equal(results.step % 10, 0)):
    tf.compat.v2.summary.scalar("log pred hmc", state_to_log_like(state, data_test, model), step=tf.cast(results.step, tf.int64))
    return ()
step_is = 0
def trace_fn_nuts(state, results):
    global step_is
    step_is +=1
    tf.summary.scalar("log pred nuts", state_to_log_like(state, data_test, model), step=step_is)
    return ()



model = Ard(num_features)
# Define the regression model

joint_log_prob2 = lambda *args: model.joint_log_prob(data_train, *args)

initial_chain_state = model.return_initial_state()
# Need to have a starting state for HMC and Nuts for chain


num_results = int(970)
# not sure why but for HMC num results must be smaller than 970.
# undergoing investigation...
num_burnin_steps = int(200)

# Defining kernels for HMC and NUTS
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=joint_log_prob2,
        num_leapfrog_steps=3,
        step_size=step_size_hmc),
    num_adaptation_steps=int(num_burnin_steps * 0.8))

nuts = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=joint_log_prob2,
    step_size=0.1,
    max_tree_depth=10,
    max_energy_diff=1000.0,
    unrolled_leapfrog_steps=1, parallel_iterations=10, seed=None, name=None)

@tf.function # tf.function creates a graph of following function.
def run_chain_hmc():
  print("running hmc chain")
  # Run the chain (with burn-in).
  states, is_accepted = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=initial_chain_state,
      kernel=adaptive_hmc,
    trace_fn=trace_fn)
  return states, is_accepted

@tf.function
def run_chain_nuts():
  print("running nuts chain")
  # Run the chain (with burn-in).
  states, is_accepted = tfp.mcmc.sample_chain(
      num_results= num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=initial_chain_state,
      kernel=nuts,
      trace_fn=trace_fn_nuts)
  return states, is_accepted

if(what_to_run == "advi"):
    bij = tfp.bijectors.Log()
    num_results = 10000
    step_size = 0.001
    # run the advi
    with summary_writer.as_default():
        theta = run_advi(model.num_params,
                         joint_log_prob2,
                         model.log_likelihood,
                         data_test,
                         bijector=bij)
        
if (what_to_run == "hmc"):
    with summary_writer.as_default():
        theta = run_chain_hmc()
    summary_writer.close()
        
if (what_to_run == "nuts"):
    with summary_writer.as_default():
        theta = run_chain_nuts()
    summary_writer.close()

print(step_is)


