from typing import List
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_legend(pred_set, output_fp, class_names: List[str], colormap: List[List[int]]):
    fig, ax = plt.subplots(figsize=(8, 4))

    legend_elements = []
    for idx in pred_set:
        name = class_names[idx]
        color = np.array(colormap[idx]) / 255
        legend_elements.append(mpatches.Patch(color=color, label=name))
    ax.legend(handles=legend_elements, loc='center')
    ax.axis('off')
    plt.savefig(output_fp, bbox_inches='tight', dpi=300)


def confuse_matrix(y: ArrayLike, x: ArrayLike, class_names: List[str], axis_names: List[str]= None, output_fp: str= None):

    cm = confusion_matrix(y, x, normalize= 'all', labels= range(37))
    cm = np.ceil(cm * 1000).astype(int)
    mask = cm == 0

    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(cm, square= True, mask= mask, annot= True, fmt='d', linewidths=0.5, linecolor='black', cmap='YlGnBu', xticklabels=class_names, yticklabels= class_names)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=45)

    if axis_names is not None:
        plt.xlabel(axis_names[0])
        plt.ylabel(axis_names[1])
    else:
        plt.ylabel('y')
        plt.xlabel('x')
    plt.title('Confusion Matrix')

    if output_fp is not None:
        plt.savefig(output_fp, bbox_inches='tight', dpi=300)
    else:
        plt.show()