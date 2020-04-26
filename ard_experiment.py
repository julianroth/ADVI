import numpy as np
import train_log as tl
from models.ard import Ard


# Making training data
def make_training_data(num_samples, dims, sigma, mu=0):
  """
  Creates training data when half of the regressors are 0
  """
  x = np.random.randn(dims, num_samples).astype(np.float64)
  w = sigma * np.random.randn(1, dims).astype(np.float64)
  noise = np.random.randn(num_samples).astype(np.float64)
  noise = 0
  w[:,:int(dims/2)] = 0.
  y = w.dot(x) + (noise/2) + mu
    
  return y, x, w


def sep_training_test(y,x,test):
  y_train = y[:,test:]
  x_train = x[:,test:]
  
  y_test = y[:,:test]
  x_test = x[:,:test]
  return y_train, y_test, x_train, x_test

num_features = 250

y, x, w = make_training_data(10000, num_features,10, mu=0)
y_train, y_test, x_train, x_test = sep_training_test(y,x,1000)

train_data = (y_train, x_train)
test_data = (y_test, x_test)

# Define and train model

model = Ard(num_features=num_features, transform=False)

# run advi
tl.run_train_advi(model, train_data, test_data, step_limit=1000, lr=0.1, adam=True)
# paper saids they used lr = 0.1
tl.run_train_advi(model, train_data, test_data, step_limit=5000, lr=0.1, adam=True, m=10)
model = Ard(num_features=num_features, transform=True)

tl.run_train_hmc(model, train_data, test_data, step_size=0.1, num_results=1000)

tl.run_train_nuts(model, train_data, test_data, step_size=0.001, num_results=1000)
