import numpy as np
import train_log as tl
from models.ard import Ard

#import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(True)
# uncomment above code when wanting to run everything in
# eager mode
# currently, the @tf.function turns those parts in to graphs.
# not sure why but needed for plotting. 

# Making training data
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

num_features = 200

y, x, w = make_training_data(1000, num_features, 2)
y_train, y_test, x_train, x_test = sep_training_test(y,x,10)

#print(y_train.shape)
#print(y_test.shape)
#print(x_train.shape)
#print(x_test.shape)

train_data = (y_train, x_train)
test_data = (y_test, x_test)

# Define and train model

model = Ard(num_features=num_features)

# run hmc
#tl.run_train_hmc(model, train_data, test_data,
#                 step_size=0.001, num_results=100, num_burnin_steps=100)

# run nuts
#tl.run_train_nuts(model, train_data, test_data,
#                  step_size=0.001, num_results=20, num_burnin_steps=0)

# run advi
tl.run_train_advi(model, train_data, test_data,
                  step_limit=200)

# good step_size for hmc is 0.1
# good step_size for advi is 0.001
