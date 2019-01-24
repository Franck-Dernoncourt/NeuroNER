'''
Miscellaneous functions for plots
'''
from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg') # http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
import matplotlib.pyplot as plt
import sklearn.preprocessing
from matplotlib import cm

def get_cmap():
    '''
    http://stackoverflow.com/questions/37517587/how-can-i-change-the-intensity-of-a-colormap-in-matplotlib
    '''
    cmap = cm.get_cmap('RdBu', 256) # set how many colors you want in color map
    # modify colormap
    alpha = 1.0
    colors = []
    for ind in range(cmap.N):
        c = []
        if ind<128 or ind> 210: continue
        for x in cmap(ind)[:3]: c.append(min(1,x*alpha))
        colors.append(tuple(c))
    my_cmap = matplotlib.colors.ListedColormap(colors, name = 'my_name')
    return my_cmap

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu', fmt="%.2f", graph_filepath='', normalize=False, remove_diagonal=False):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857
    - http://stackoverflow.com/a/25074150/395857
    '''
    if normalize:
        AUC = sklearn.preprocessing.normalize(AUC, norm='l1', axis=1)
        
    if remove_diagonal:
        matrix = np.copy(AUC)
        np.fill_diagonal(matrix, 0)
        if len(xticklabels)>2:
            matrix[:,-1] = 0
            matrix[-1, :] = 0
        values= matrix.flatten()
    else:
        values = AUC.flatten()
    vmin = values.min()
    vmax = values.max()

    # Plot it out
    fig, ax = plt.subplots()
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=get_cmap(), vmin=vmin, vmax=vmax)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c, fmt=fmt)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))

    if graph_filepath != '':
        plt.savefig(graph_filepath, dpi=300, format='png', bbox_inches='tight')
        plt.close()


def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu', from_conll_json=False):
    '''
    Plot scikit-learn classification report.
    Extension based on http://stackoverflow.com/a/31689645/395857
    '''
    classes = []
    plotMat = []
    support = []
    class_names = []
    if from_conll_json:
        for label in sorted(classification_report.keys()):
            support.append(classification_report[label]["support"])
            classes.append('micro-avg' if label=='all' else label)
            class_names.append('micro-avg' if label=='all' else label)
            plotMat.append([float(classification_report[label][x]) for x in ["precision", "recall", "f1"]]) 
    else:
        lines = classification_report.split('\n')
        for line in lines[2 : (len(lines) - 1)]:
            t = line.strip().replace('avg / total', 'micro-avg').split()
            if len(t) < 2: continue
            classes.append(t[0])
            v = [float(x)*100 for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            plotMat.append(v)
    
    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = True

    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)


def plot_hist(sequence, xlabel, ylabel, title, graph_path):
    xmin = min(sequence)
    xmax = max(sequence)
    step = 1
    y, x = np.histogram(sequence, bins=np.linspace(xmin, xmax, (xmax-xmin+1)/step))

    plt.bar(x[:-1], y, width=x[1]-x[0], color='red', alpha=0.5)
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=8)
    plt.title(title, fontsize=12)
    plt.ylabel(ylabel, fontsize=8)
    plt.savefig(graph_path, dpi=300, format='png', bbox_inches='tight')
    plt.close()


def plot_barh(x, y, xlabel, ylabel, title, graph_path):
    width = 1
    fig, ax = plt.subplots()
    ind = np.arange(len(y))  # the x locations for the groups
    ax.barh(ind, y, color="blue")
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(x, minor=False)
    # http://stackoverflow.com/questions/30228069/how-to-display-the-value-of-the-bar-on-each-bar-with-pyplot-barh/30229062#30229062
    for i, v in enumerate(y):
        ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(graph_path, dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
    plt.clf()
    plt.close()


def plot_precision_recall_curve(recall, precision, graph_path, title):
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(graph_path, dpi=600, format='pdf', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
    plt.close()


def plot_roc_curve(fpr, tpr, graph_path, title):
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(graph_path, dpi=600, format='pdf', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
    plt.close()


def plot_threshold_vs_accuracy_curve(accuracies, thresholds, graph_path, title):
    plt.clf()
    plt.plot(thresholds, accuracies, label='ROC curve')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(graph_path, dpi=600, format='pdf', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
    plt.close()


