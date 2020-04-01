import train_log as tl
from models.constrained_gamma_poisson import Gamma_Poisson

# generate dirichlet exponential model
model = Gamma_Poisson()

# run advi
tl.run_train_advi(model, model._train_data, model._test_data,
                  lr=0.1)

#tl.run_train_advi(model, model._train_data, model._test_data,
#                  lr=0.1, m=10)

#run hmc
#tl.run_train_hmc(model, model._train_data, model._test_data,
#                 step_size=0.001, num_results=500, num_burnin_steps=100)

#tl.run_train_nuts(model, model._train_data, model._test_data,
#                 step_size=0.001, num_results=500, num_burnin_steps=100)
