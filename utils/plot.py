import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
import glob


def plot_results(advi_file, advi_file_2=None, hmc_file=None, nuts_file=None,
         time_log_scale=True, y_lim=None, save_file=None):
    """
    Plots graph from log files

    Parameters
    ----------
    advi_file : str, path object or file-like object
        CSV file of advi results
        Requires label, time, value columns with labels 'log pred advi'
    advi_file_2 : str, path object or file-like object
        optional CSV file of more advi results
        If both advi files passed to the function, we assume M=1 for the first and M=10 for the second
    hmc_file : str, path object or file-like object
        optional CSV file of hmc results
        Requires label, time, value columns with labels 'log pred hmc'
    nuts_file : str, path object or file-like object
        optional CSV file of nuts results
        Requires label, time, value columns with labels 'log pred nuts'
    time_log_scale : boolean
        time x axis on log scale
    save_file : str
        optional file name for saving the plots, format (e.g. png or jpeg) will be inferred from file name
    """
    plt.clf()
    labels = []
    colors = []
    advi_df = pd.read_csv(advi_file)
    if advi_file_2 is not None:
        data = create_dataframe(advi_df, 'advi', 'ADVI (M = 1)')
        advi_df_2 = pd.read_csv(advi_file_2)
        data = data.append(create_dataframe(advi_df_2, 'advi', 'ADVI (M = 10)'), ignore_index=True)
        labels += ['ADVI (M = 1)', 'ADVI (M = 10)']
        colors += ["#9b59b6", "#3498db"]
    else:
        data = create_dataframe(advi_df, 'advi')
        labels += ['ADVI']
        colors += ["#9b59b6"]
    if hmc_file is not None:
        hmc_df = pd.read_csv(hmc_file)
        data = data.append(create_dataframe(hmc_df, 'hmc'), ignore_index=True)
        labels += ['HMC']
        colors += ["#95a5a6"]
    if nuts_file is not None:
        nuts_df = pd.read_csv(nuts_file)
        data = data.append(create_dataframe(nuts_df, 'nuts'), ignore_index=True)
        labels += ['NUTS']
        colors += ["#e74c3c"]

    # style of graph
    sns.set_style("ticks", {'axes.grid': False,
                            'axes.spines.right': False,
                            'axes.spines.top': False,
                            'axes.spines.bottom': False})
    ax = plt.gca()
    ax.yaxis.grid(True)
    ax.legend(loc='upper left', frameon=False)
    grid = sns.lineplot(data=data, x='Seconds', y='Average Log Predictive', hue='Label',
                        style='algorithm', palette=sns.color_palette(colors), legend='full')
    if time_log_scale:
        grid.set(xscale="log")
    leg = plt.legend(title='', loc='lower right', labels=labels)
    leg.get_frame().set_linewidth(0.0)
    
    if y_lim is not None:
        grid.set_ylim(y_lim[0], y_lim[1])
    
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, dpi=200)


def create_dataframe(df, algorithm, label=None):
    if label is None:
        label = algorithm.upper()
    time = df['time']
    value = df.loc[df['label'] == 'avg log pred {}'.format(algorithm)]['value']
    data = pd.DataFrame({'Seconds' : time,
                         'Average Log Predictive' : value,
                         'algorithm' : algorithm.upper(),
                         'Label' : label})
    return data


def plot(files, algorithms, labels, time_log_scale=True, save_file=None):
    """
    :param files: List of str, path object or file-like objects
        List of CSV files of results
    :param algorithms: List of str
        Algorithms used to extract values from csv files (advi, hmc or nuts)
    :param labels: List of str
        Labels for legend
    :param time_log_scale: same as plot_results
    :param save_file:  same as plot_results
    """
    plt.clf()
    num = len(files)
    data = None
    for i in range(num):
        df = pd.read_csv(files[i])
        plot_df = create_dataframe(df, algorithms[i], labels[i])
        if data is None:
            data = plot_df
        else:
            data = data.append(plot_df)

    sns.set_style("ticks", {'axes.grid': False,
                            'axes.spines.right': False,
                            'axes.spines.top': False,
                            'axes.spines.bottom': False})
    ax = plt.gca()
    ax.yaxis.grid(True)
    ax.legend(loc='upper left', frameon=False)
    grid = sns.lineplot(data=data, x='Seconds', y='Average Log Predictive', hue='Label',
                        style='algorithm', legend='full')
    if time_log_scale:
        grid.set(xscale="log")
    leg = plt.legend(title='', loc='lower right', labels=labels)
    leg.get_frame().set_linewidth(0.0)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, dpi=150)


if __name__ == '__main__':
    #plot('~/ADVI/logs/20200401-093255_advi.csv', advi_file_2='~/ADVI/logs/20200401-093323_advi.csv', hmc_file='~/ADVI/logs/20200401-093545_hmc.csv')
    #plot('~/ADVI/logs/20200401-103711_advi.csv', advi_file_2='~/ADVI/logs/20200401-103928_advi.csv', hmc_file='~/ADVI/logs/20200401-141630_hmc.csv', nuts_file='~/ADVI/logs/20200401-142000_nuts.csv')
    #plot_results('~/ADVI/logs/20200403-145207_advi.csv', advi_file_2='~/ADVI/logs/20200403-150123_advi.csv', hmc_file='~/ADVI/logs/20200403-153542_hmc.csv', nuts_file='~/ADVI/logs/20200403-153853_nuts.csv')
    #plot_results('~/ADVI/logs/20200404-103938_advi.csv', advi_file_2='~/ADVI/logs/20200404-110728_advi.csv',hmc_file='~/ADVI/logs/20200404-143631_hmc.csv',nuts_file='~/ADVI/logs/20200404-170222_nuts.csv')
    #plot_results('~/ADVI/logs/20200404-222551_advi.csv', advi_file_2= '~/ADVI/logs/ard_plot_fin2/20200404-103938_advi.csv', hmc_file='~/ADVI/logs/20200405-081706_hmc.csv')
    #plot_results('~/ADVI/logs/20200406-174903_advi.csv', hmc_file='~/ADVI/logs/20200406-184211_hmc.csv', nuts_file='~/ADVI/logs/20200406-201903_nuts.csv')
    import os
    directory = '/users/Mizunt/ADVI/logs/'
    os.chdir(directory)
    sorted_dir = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
    print(sorted_dir[-2]) 
    print(sorted_dir[-1])
    plot_results(sorted_dir[-9], hmc_file=sorted_dir[-1])
