from models import constrained_gamma_poisson as cgp
from models import dirichlet_exponential as dem 
from data import frey_face  
import tensorflow as tf
from train_plot import run_train
# prepare unnormalised log probability function
data = frey_face.load_data()
#target_log_prob = lambda *args: dem.log_posterior_sampling(data, *args)

#init_state = dem.initial_state()
#step_size = dem.std_step_size()
#n = (dem.U + dem.I) * dem.K
theta = cgp.theta_prior_distribution().sample(tf.constant(cgp.U * cgp.K, dtype=tf.int64))
beta = cgp.beta_prior_distribution().sample(tf.constant(cgp.K * cgp.I, dtype=tf.int64))
init_state = tf.concat([theta, beta], 0)
step_size = tf.concat([tf.reshape(cgp.stddev_theta(), [-1]), tf.reshape(cgp.stddev_beta(), [-1])], 0)
n = (28 + 20) * 1
# U I and K would be better if its passed in to the model.
# currently they are defined twice
# i.e. should be model = DE(28, 20, 10)

model = dem.DirichletExponential()
run_train(model, "hmc", 0.001, train_data,
          test_data, num_results=500, num_burnin_steps=100)
# good step_size for hmc is 0.1
# good step_size for advi is 0.001                                                                                                                                                   


