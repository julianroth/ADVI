import train_log as tl
from models.dirichlet_exponential import DirichletExponential
from data import frey_face


# prepare data for training and testing
data = frey_face.load_data()
held_out = 100
train_data = data[:-held_out, :]
test_data = data[-held_out:, :]

# generate dirichlet exponential model
model = DirichletExponential(users=28, items=20, factors=10)

# run advi
tl.run_train_advi(model, train_data, test_data,
                  step_limit=500)

# run hmc
#tl.run_train_hmc(model, train_data, test_data,
#                 step_size=0.001, num_results=500, num_burnin_steps=100)
