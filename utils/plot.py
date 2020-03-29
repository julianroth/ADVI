import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd

def plot(advi_file, advi_file_2=None, hmc_file=None, nuts_file=None, time_log_scale=True, save_file=None):
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
        optional file name for saving the plots
    """
    # style of graph
    colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"]
    sns.set_style("ticks", {'axes.grid' : False,
                            'axes.spines.right' : False,
                            'axes.spines.top' : False,
                            'axes.spines.bottom' : False})
    ax = plt.gca()
    ax.yaxis.grid(True)
    ax.legend(loc='upper left', frameon=False)
    print(sns.axes_style())

    labels = []
    advi_df = pd.read_csv(advi_file)
    if advi_file_2 is not None:
        data = create_dataframe(advi_df, 'advi', 'ADVI (M = 1)')
        advi_df_2 = pd.read_csv(advi_file_2)
        data = data.append(create_dataframe(advi_df_2, 'advi', 'ADVI (M = 10)'), ignore_index=True)
        labels += ['ADVI (M = 1)', 'ADVI (M = 10)']
    else:
        data = create_dataframe(advi_df, 'advi')
        labels += ['ADVI']
    if hmc_file is not None:
        hmc_df = pd.read_csv(hmc_file)
        data = data.append(create_dataframe(hmc_df, 'hmc'), ignore_index=True)
        labels += ['HMC']
    if nuts_file is not None:
        nuts_df = pd.read_csv(nuts_file)
        data = data.append(create_dataframe(nuts_df, 'nuts'), ignore_index=True)
        labels += ['NUTS']

    grid = sns.lineplot(data=data, x='Seconds', y='Average Log Predictive', hue='Label',
                        style='algorithm', palette=sns.color_palette(colors), legend='full')
    if time_log_scale:
        grid.set(xscale="log")
    leg = plt.legend(title='', loc='lower right', labels=labels)
    print(leg)
    leg.get_frame().set_linewidth(0.0)
    plt.show()


def create_dataframe(df, algorithm, label=None):
    if label is None:
        label = algorithm.upper()
    time = df['time']
    value = df.loc[df['label'] == 'log pred {}'.format(algorithm)]['value']
    data = pd.DataFrame({'Seconds' : time,
                         'Average Log Predictive' : value,
                         'algorithm' : algorithm.upper(),
                         'Label' : label})
    return data


sample_logs_directory = 'utils/sample_logs/'
plot(sample_logs_directory + 'advi.csv', sample_logs_directory + 'advi_2.csv',
     sample_logs_directory + 'hmc.csv', sample_logs_directory + 'nuts.csv')