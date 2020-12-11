'''
sdss_model_evaluate - set of functions to evaluate sdss regression model 
(i.e. input RGB image, output continuous scalar 0-1).
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_regression(y_test, y_predicted, scatter=True, color='dodgerblue', label=None, ax=None):
    '''
    plots regression between test and predicted values to evaluate
    model. returns scatter plot of data along with binned mean values
    (with scatter errorbars) to compare to 1 to 1 relationship.
    
    ----------
    Parameters
    
    y_test : np.array()
        Array of test (numeric) values
        
    y_predicted : np.array()
        Array of predicted y values
    
    scatter : boolean
        Whether to include raw data scatter or not
        
    color : str
        Colour of 
    
    ax : plt.matplotlib object
    
    -------
    Returns
    
    '''
    # setting overall linewidth for plotting
    linewidth = 4
    
    if ax is None:
        fig = plt.figure(figsize=(5, 5) )
        ax = fig.add_subplot()
        
    # plotting 1 to 1 
    x = np.linspace(0, 1, 20)
    ax.plot(x, x, linestyle='dashed', color='k', lw=linewidth)
    
    # plotting scatter of data
    if scatter == True:
        ax.scatter(y_test, y_predicted, color='salmon', alpha=0.1)
    
    # binned mean and std
    bin_means, bin_edges, binnumber = stats.binned_statistic(y_test, y_predicted, statistic='mean')
    bin_std, _, _ = stats.binned_statistic(y_test, y_predicted, statistic='std')
    
    # finding bin centres (x)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    
    # plotting bin width (x) against std (y)
    ax.hlines(bin_means, bin_edges[:-1], bin_edges[1:], color=color, lw=linewidth, label=label)
    ax.errorbar(bin_centers, bin_means, yerr=bin_std, color=color, lw=linewidth)

    ax.set_xlabel('Actual values', fontsize=15)
    ax.set_ylabel('Predicted values', fontsize=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return ax

