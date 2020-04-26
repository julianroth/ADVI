import train_log as tl
from models.mixed_gauss import MixedGauss
from utils.plot import plot, plot_results
import sys


if __name__ == "__main__":
    arg = str(sys.argv[-1])

    if arg == 'plot':
        base = 'logs/mixed_gauss/mg_'
        advi_file = base + 'advi.csv'

        plot_results(advi_file)
    else:
        model = MixedGauss(id_transform=False)
        # run advi
        advi = tl.run_train_advi(model, None, None,
                          step_limit=-1, m=1, lr=0.1)

        import tensorflow as tf
        import tensorflow_probability as tfp
        tfd = tfp.distributions

        num_samples = 100000
        d1 = tfd.MultivariateNormalFullCovariance(model._mu[0, :], model._std[0, :, :])
        d2 = tfd.MultivariateNormalFullCovariance(model._mu[1, :], model._std[1, :, :])
        data = tf.concat([d1.sample(num_samples // 2), d2.sample(num_samples // 2)], 0)
        print(data.shape)
        x = data[:, 0]
        y = data[:, 1]
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        sns.kdeplot(x.numpy(), y.numpy(), shade=False, shade_lowest=False, cmap='Greens', legend=True)

        d = tfd.Normal(advi.mu, tf.math.exp(advi.omega))
        data = model.bijector().inverse(d.sample(num_samples))
        x = data[:, 0]
        y = data[:, 1]
        label_patch_1 = mpatches.Patch(
            color=sns.color_palette('Greens')[2],
            label='Target Distribution')
        sns.kdeplot(x.numpy(), y.numpy(), shade=False, shade_lowest=False, cmap='Reds')

        label_patch_2 = mpatches.Patch(
            color=sns.color_palette('Reds')[2],
            label='Variational Distribution')

        plt.legend(handles=[label_patch_1, label_patch_2], loc='lower right')
        plt.savefig('mg_sh.png', dpi=200)