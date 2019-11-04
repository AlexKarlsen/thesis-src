#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from tqdm import tqdm

params = {'legend.fontsize': 'large',
          'figure.figsize': (16, 9),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
plt.rcParams.update(params)


models = ['b-densenet', 'b-resnet', 'msdnet']
tests = ['confidence', 'score-margin']
nets = { model : {test : {} for test in tests} for model in models}
nets['b-densenet']['confidence'] = pd.read_csv('densenet100_confidence.csv', index_col=0)
nets['b-densenet']['score-margin'] = pd.read_csv('densenet100_score_margin.csv', index_col=0)

nets['b-resnet']['confidence'] = pd.read_csv('resnet101_confidence1.csv', index_col=0)
nets['b-resnet']['score-margin'] = pd.read_csv('resnet101_score_margin1.csv', index_col=0)

nets['msdnet']['confidence'] = pd.read_csv('msdnet_confidence1.csv', index_col=0)
nets['msdnet']['score-margin'] = pd.read_csv('msdnet_score_margin1.csv', index_col=0)

nets['msdnet']['score-margin']


def analyze_df(models, exits):
    tests = ['confidence', 'score-margin']
    data = {
        model : {
            test : {
                'exited' : {},
                'correct' : {},
                'incorrect': {},
                'accuracy': {}
            }
            for test in tests
        }
        for model in models
    }
    
    for model, n in tqdm(zip(models, exits), leave=False, unit='tests'):
        for test in models[model]:
            for exit in range(n):
                exited = []
                accuracy = []
                correct = []
                incorrect = []
                for t in np.arange(0.1, 1, 0.1):
                    n_exited = len(models[model][test].loc[(models[model][test]['threshold'] == t) 
                                             & (models[model][test]['exit'] == exit)
                                             & (models[model][test]['exited']==1)])
                    exited.append(n_exited)
                    
                    correct.append((len(models[model][test].loc[(models[model][test]['threshold'] == t) 
                                                            & (models[model][test]['exit'] == exit) 
                                                            & (models[model][test]['correct'] == 1) 
                                                            & (models[model][test]['exited']==1)])))
                    incorrect.append((len(models[model][test].loc[(models[model][test]['threshold'] == t) 
                                                    & (models[model][test]['exit'] == exit) 
                                                    & (models[model][test]['correct'] == 0) 
                                                    & (models[model][test]['exited']==1)])))
                    
                    accuracy.append(len(models[model][test].loc[(models[model][test]['threshold'] == t) 
                                                  & (models[model][test]['exit'] == exit) 
                                                  & (models[model][test]['exited']==1) 
                                                  & (models[model][test]['correct']==1)])/n_exited)
                    
                data[model][test]['exited']['exit-{}'.format(exit)] = exited
                #th_exit_acc.append(exit_n_acc)
                #th_false_if_exited.append(exit_n_false)
                data[model][test]['accuracy']['exit-{}'.format(exit)] = accuracy
                #th_time.append(exit_time)
                data[model][test]['correct']['exit-{}'.format(exit)]  = correct
                data[model][test]['incorrect']['exit-{}'.format(exit)] = incorrect

    
    return data
    



output = analyze_df(nets, exits=[4, 4, 5])

output

n_samples = 5000
n_thresholds = np.arange(9)

fig, ax = plt.subplots(1,1, figsize=(16,9), sharex=True, sharey=True)

color_palette = sns.color_palette()

plt.setp(ax, xticks=n_thresholds, xticklabels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

for model in tqdm(output):
    for test in output[model]:
        for exit in range(len(output[model][test]['correct'])):
            accuracy = np.array(output[model][test]['accuracy']['exit-{}'.format(exit)])
            correct = np.array(output[model][test]['correct']['exit-{}'.format(exit)])
            incorrect = np.array(output[model][test]['incorrect']['exit-{}'.format(exit)])
            not_exited = n_samples - (correct + incorrect) 

            # normalizing
            correct = correct / n_samples
            incorrect = incorrect / n_samples
            not_exited = not_exited / n_samples

            ax.bar(n_thresholds, correct, color=color_palette[2])
            ax.bar(n_thresholds, incorrect, bottom = correct, color=color_palette[3])
            ax.bar(n_thresholds, not_exited, bottom = correct + incorrect, color=color_palette[0])
            #ax.plot(n_thresholds, accuracy)

            ax.set_title('Exit-{}: {}'.format(exit, test))
            ax.set_ylim([0,1])
            ax.set(xlabel='threshold', ylabel='frequency')
            fig.tight_layout()
            #fig.subplots_adjust(left=0.15, top=0.95)
            #plt.show()
            plt.savefig('threshold_analysis_{}_{}_{}.png'.format(model, test, exit))

print('plotting complete')

