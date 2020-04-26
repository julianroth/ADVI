import train_log as tl
from models.constrained_gamma_poisson import Gamma_Poisson


# generate dirichlet exponential model
model = Gamma_Poisson(transform=False)

# run advi
tl.run_train_advi(model, model._train_data, model._test_data, lr=0.1, step_limit=38000, adam=True)
# paper says they used lr = 0.1
tl.run_train_advi(model, model._train_data, model._test_data, lr=0.1, m=10, step_limit=13000, adam=True)
model = Gamma_Poisson(transform=True)

tl.run_train_hmc(model, model._train_data, model._test_data, step_size=0.001, num_results=12000, num_burnin_steps=0)

tl.run_train_nuts(model, model._train_data, model._test_data, step_size=0.001, num_results=12000, num_burnin_steps=0)
