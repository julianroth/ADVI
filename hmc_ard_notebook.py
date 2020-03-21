
# coding: utf-8

# In[1]:

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from advi.model import ADVIModel
from models.ard import Ard
#from models.ard import joint_log_prob, return_initial_state, sep_params, log_likelihood, some_kind_of_loss
# NEXT TO DO MUST CHANGE INITAL CHAIN STATES TO SAMPLE FROM DISTRIBUTION
# Target distribution is proportional to: `exp(-x (1 + x))`.

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

num_features = 10

y, x, w = make_training_data(100, num_features, 2)
y_train, y_test, x_train, x_test = sep_training_test(y,x,10)

print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)
data_train = (y_train, x_train)
data_test = (y_test, x_test)


def state_to_log_like(states, data):
    sample_mean = tf.reduce_mean(states, axis=[0])
    return model.log_likelihood(data, states)

def state_to_loss(states, data):
    sample_mean = tf.reduce_mean(states, axis=[0])
    w, _, _ = sep_params(states, num_features)
    return model.loss(data, w)


# ## Defining summary writer
summary_writer = tf.compat.v2.summary.create_file_writer('/tmp/summary_chain', flush_millis=10000)
    
def trace_fn(state, results):
  with tf.compat.v2.summary.record_if(tf.equal(results.step % 10, 0)):
    tf.compat.v2.summary.scalar("log pred hmc", state_to_log_like(state, data_test), step=tf.cast(results.step, tf.int64))
    return ()
step_is = 0
def trace_fn_nuts(state, results):
    global step_is
    step_is +=1
    tf.summary.scalar("log pred nuts", state_to_log_like(state, data_test), step=step_is)
    return ()


# # Define the regression model
model = Ard(num_features)
# Need to have a starting state for HMC and Nuts for chain
joint_log_prob2 = lambda *args: model.joint_log_prob(data_train, *args)
initial_chain_state = model.return_initial_state()


# ## Defining kernels for HMC and NUTS

# In[6]:

num_results = int(10000)
num_burnin_steps = int(800)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=joint_log_prob2,
        num_leapfrog_steps=3,
        step_size=0.1),
    num_adaptation_steps=int(num_burnin_steps * 0.8))

nuts = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=joint_log_prob2,
    step_size=10,
    max_tree_depth=10,
    max_energy_diff=1000.0,
    unrolled_leapfrog_steps=1, parallel_iterations=10, seed=None, name=None)


# In[7]:

@tf.function
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


# In[8]:

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
    


# ## ADVI Stuff starts here

# ### Functions to pass in to summary writer

# In[9]:

def get_ll_from_advi(y_test, x_test, advi, regression_model):
    theta_intermediate = advi.sample()
    theta_intermediate = tf.squeeze(theta_intermediate)
    likelihood = regression_model.log_likelihood(data_test, theta_intermediate)
    return likelihood

def get_loss_from_advi(y_test, x_test, advi, regression_model):
    theta_intermediate = advi.sample()
    theta_intermediate = tf.squeeze(theta_intermediate)
    w, _, _ = regression_model.sep_params(theta_intermediate)
    return regression_model.loss(y_test, x_test, w)


# In[10]:

def run_advi(nsteps, step_size, dim, log_prob, bijector, m, regression_model):
        advi = ADVIModel(dim, log_prob, bijector, m)
        for t in range(nsteps):
            grad_mu, grad_omega = advi.gradients()
            advi.mu = tf.add(advi.mu, step_size * grad_mu)
            advi.omega = tf.add(advi.omega, step_size * grad_omega)
            tf.summary.scalar('elbo', advi.elbo(), step=t)
            tf.summary.scalar('log likelihood', get_ll_from_advi(y_test, x_test, advi, regression_model), step=t)
            tf.summary.scalar('loss', get_loss_from_advi(y_test, x_test, advi, regression_model), step=t)
            if(t%1 == 0):
                theta_intermediate = advi.sample()
                theta_intermediate = tf.squeeze(theta_intermediate)
                wpred, _, _ = model.sep_params(theta_intermediate)
                loss = regression_model.loss(y_test, x_test, wpred)
                likelihood = regression_model.log_likelihood(data_test, theta_intermediate)
                elbo_loss = advi.elbo()
                print("weights", wpred)
                print("loss", loss)
                print("elbo", elbo_loss)
        theta = advi.sample()
        return theta


# In[11]:

# target function 

what_to_run = "advi"
if(what_to_run == "advi"):
    bij = tfp.bijectors.Log()
    num_results = 10000
    step_size = 0.001
    # run the advi
    with summary_writer.as_default():
        theta = run_advi(num_results,
                         step_size,
                         model.num_params,
                         joint_log_prob2,
                         bij,
                         10, model)
        
if (what_to_run == "hmc"):
    with summary_writer.as_default():
        theta = run_chain_hmc()
    summary_writer.close()
        
if (what_to_run == "nuts"):
    with summary_writer.as_default():
        theta = run_chain_nuts()
    summary_writer.close()

print(step_is)


