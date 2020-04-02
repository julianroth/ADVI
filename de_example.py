import train_log as tl
from models.dirichlet_exponential import DirichletExponential
from data import frey_face
from utils.sep_data import sep_training_test

# prepare data for training and testing
data = frey_face.load_data()
held_out = 0.2
train_data, test_data = sep_training_test(data, test_split=held_out, permute=True)

# generate dirichlet exponential model
model = DirichletExponential(users=28, items=20, factors=10)

# run advi
tl.run_train_advi(model, train_data, test_data,
                  m=1, p=1, skip_steps=10)
