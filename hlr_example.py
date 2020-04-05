import train_log as tl
from models.hlr import HLR
from utils.plot import plot, plot_results
import sys


if __name__ == "__main__":
    # generate hierarchical logistic regression model
    model = HLR(permute=True)
    arg = str(sys.argv[-1])

    if arg == 'plot':
        base = 'logs/hlr/hlr_'
        advi_file = base + 'advi_m_1.csv'
        advi_file_2 = base + 'advi_m_10.csv'
        advi_file_2 = base + 'advi_adam_4.csv'
        hmc_file = base + 'hmc.csv'
        nuts_file = base + 'nuts.csv'

        plot_results(advi_file, advi_file_2)#, hmc_file, nuts_file)
    else:
        # run advi
        tl.run_train_advi(model, model._train_data, model._test_data,
                          step_limit=-1, m=1, lr=0.3, adam=True)
        sys.exit(1)
        tl.run_train_advi(model, model._train_data, model._test_data,
                          step_limit=-1, m=10, lr=0.1)

        # run hmc
        tl.run_train_hmc(model, model._train_data, model._test_data,
                         step_size=0.001, num_results=1000, num_burnin_steps=0)

        # run nuts
        tl.run_train_nuts(model, model._train_data, model._test_data,
                         step_size=0.001, num_results=100, num_burnin_steps=0)
