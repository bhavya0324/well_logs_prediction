import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot(cm, classifier):
    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix with labels for '+classifier)
    ax.set_xlabel('\nPredicted log category')
    ax.set_ylabel('Actual log category')
    ax.xaxis.set_ticklabels(['Cased Hole', 'LWD', 'Open Hole'])
    ax.yaxis.set_ticklabels(['Cased Hole', 'LWD', 'Open Hole'])
    plt.show()