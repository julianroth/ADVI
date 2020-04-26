import train_log as tl
from models.hlr import HLR
from utils.plot import plot, plot_results
import sys


if __name__ == "__main__":
    # generate hierarchical logistic regression model
    model = HLR(permute=True, num_test=1566)
    arg = str(sys.argv[-1])

    if arg == 'plot':
        base = 'logs/hlr/hlr_'
        advi_file = base + 'advi_adam_m_1.csv'
        advi_file_2 = base + 'advi_adam_m_10.csv'
        hmc_file = base + 'hmc.csv'
        hmc_t_file = base + 'hmc_transform.csv'
        nuts_file = base + 'nuts.csv'
        nuts_t_file = base + 'nuts_transform.csv'

        plot_results(advi_file, advi_file_2, hmc_t_file, nuts_t_file, save_file='hlr.png')
        #plot([hmc_file, hmc_t_file, nuts_file, nuts_t_file], ['hmc', 'hmc', 'nuts', 'nuts'], ['hmc', 'hmc unconstrained', 'nuts', 'nuts unconstrained'])
        #plot(['logs/20200410-115032_hmc.csv', 'logs/20200410-115041_hmc.csv'], ['hmc', 'hmc'], ['hmc', 'hmc unconstrained'])
    else:
        # run advi
        tl.run_train_advi(model, model._train_data, model._test_data,
                          step_limit=-1, m=1, lr=0.3, adam=True)
        tl.run_train_advi(model, model._train_data, model._test_data,
                          step_limit=-1, m=10, lr=0.3, adam=True)

        # run hmc
        tl.run_train_hmc(model, model._train_data, model._test_data,
                         step_size=0.01, num_results=2500, num_burnin_steps=0, transform=True)

        # run nuts
        tl.run_train_nuts(model, model._train_data, model._test_data,
                         step_size=0.01, num_results=150, num_burnin_steps=0, transform=True)
