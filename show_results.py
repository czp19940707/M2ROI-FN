import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    method_list = [i.strip() for i in open(r'show_results.txt', 'r')]
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    color_selected = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf', 'dodgerblue', '#CC3299', '#A68064', '#215E21', '#70DB93']
    frame_save = pd.DataFrame(columns=['version', 'acc', 'sen', 'spe', 'auc'])
    for nums, method in enumerate(method_list):
        pred = np.load(os.path.join('results', method, 'pred.npy'), allow_pickle=True)
        prob = np.load(os.path.join('results', method, 'prob.npy'), allow_pickle=True)
        cls = np.load(os.path.join('results', method, 'cls.npy'), allow_pickle=True)
        fpr, tpr, threshold = roc_curve(cls, prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        cm = confusion_matrix(cls, pred)
        tp, fn, fp, tn = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        frame_save.loc[nums, 'version'] = method
        # frame_save.loc[nums, 'acc'] = round(accuracy, 5) * 100
        # frame_save.loc[nums, 'sen'] = round(sensitivity, 5) * 100
        # frame_save.loc[nums, 'spe'] = round(specificity, 5) * 100
        # frame_save.loc[nums, 'auc'] = round(roc_auc, 5) * 100

        frame_save.loc[nums, 'acc'] = str(accuracy * 100)[:5]
        frame_save.loc[nums, 'sen'] = str(sensitivity * 100)[:5]
        frame_save.loc[nums, 'spe'] = str(specificity * 100)[:5]
        frame_save.loc[nums, 'auc'] = str(roc_auc * 100)[:5]

        # plt.plot(fpr, tpr, label=method.split('.')[0].split('_ridge_')[-1] + '_' + method.split('.')[1].split('_')[0] + '_' + method.split('.')[-1].split('_')[0] + ': {}'.format(str(roc_auc)[:6]), lw=2, alpha=.8, color=color_selected[nums])
        plt.plot(fpr, tpr,
                 label=method.split('_')[-1] + ': {}'.format(str(roc_auc)[:6]), lw=2, alpha=.8,
                 color=color_selected[nums])

    frame_save.to_csv(r'result_{}.csv'.format(method.split('_')[0] + '_' + method.split('_')[1]), index=False)

    plt.title(method.split('_')[0] + '_' + method.split('_')[1])
    plt.legend(loc='lower right', ncol=2)
    plt.savefig('roc_{}.jpg'.format(method.split('_')[0] + '_' + method.split('_')[1]))
    plt.close()
