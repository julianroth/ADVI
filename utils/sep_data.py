import math
import numpy as np

# seperates x & y into training and test set
def sep_training_test(x, y=None, num_test=-1, test_split=0.2, permute=False):
  if num_test == -1:
      assert test_split >= 0 and test_split <= 1, "test split has to be between 0 and 1"
      num_test = math.floor(x.shape[0] * test_split)

  if permute:
      x = np.random.permutation(x)
      if y is not None:
          y = np.random.permutation(y)

  x_train = x[num_test:,:]
  x_test = x[num_test:,:]
  if y is not None:
      y_train = y[:num_test]
      y_test = y[:num_test]
      return (x_train, y_train), (x_test, y_test)
  return x_train, x_test
