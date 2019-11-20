import pandas as pd
import numpy as np
import seaborn as sns; sns.set('talk')
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'large',
          'figure.figsize': (16, 9),
         'axes.labelsize': 'medium',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small'}
plt.rcParams.update(params)
import json
from pandas.io.json import json_normalize

def MultiHotEncode(_labels, _scores, n_classes=100):
    '''Return topk labels and scores as multihot encoded vector. 
    Missing labels are padded with minus one.
    Missing scores are padded with zeros'''
    retLabels, retScores = [], []
    
    # dimensionality check
    if not len(_labels.shape) == len(_scores.shape):
        raise Exception('Dimensionality must match')
    
    # if only one dimensio
    if len(_labels.shape) == 1:
        multihotlabels = np.negative(np.ones(n_classes))
        multihotscores = np.zeros(n_classes)
        for l, s in zip(_labels, _scores):
            #print(l)
            multihotlabels[l] = l
            multihotscores[l] = s
        retLabels.append(multihotlabels)
        retScores.append(multihotscores)
    
    # mulitple dimension
    else:  
        for n in range(_labels.shape[0]):
            multihotlabels = np.negative(np.ones(n_classes))
            multihotscores = np.zeros(n_classes)
            for l, s in zip(_labels[n], _scores[n]):
                #print(l)
                multihotlabels[l] = l
                multihotscores[l] = s
            retLabels.append(multihotlabels)
            retScores.append(multihotscores)
    return retLabels, retScores


def Confidence(_labels, _scores, selection='additive', weights = None, n_classes=None):
    '''
    Return the confidence score prediction as an addition of prediction or max selection.
    If weights are given each prediction score are weighted accordingly. 
    If sparse prediction vector are provided and prediction labels are not consistent,
    expansion to full dimensionality is performed.
    '''
    
    # transform to numpy arrays
    _labels = np.array(_labels)
    _scores = np.array(_scores)
    
    # weight prediction by multiplying weight vector
    if weights:
        weights = np.array(weights)[:_scores.shape[0]]
        _scores = np.multiply(_scores, weights[:, None])

    # if sparse labels does not match transformation to original class space is required
    if n_classes and n_classes != _labels.shape[0]:
            MultiHotEncode(_labels, _scores, n_classes)
    
    # Sum across predtion or take max
    if selection == 'additive':
        _scores = _scores.sum(axis=0)
    elif selection == 'max':
        _scores = _scores.max(axis=0)
    else:
        raise Exception("Selection is required. Either 'additive' or 'max'")

    _scores = _scores[_scores != 0]
    _labels = _labels[_labels != -1]

    _labels = np.unique(_labels)
    
    # find best prediction
    idx = np.argmax(_scores)
    best = (_labels[idx], _scores[idx])
    
    return best

def ScoreMargin(_labels, _scores, selection='additive', weights = None):
    '''
    Return the score-margin prediction as a addition of provided predictions or a max selection.
    If weights are given each prediction score are weighted accordingly. 
    '''
    
    # transform to numpy arrays
    _labels = np.array(_labels)
    _scores = np.array(_scores)
    
    # weight prediction by multiplying weight vector
    if weights:
        weights = np.array(weights)[:_scores.shape[0]]
        _scores = np.multiply(_scores, weights[:, None])

    # create list of labels from prediction with no duplicates and corresponding score list
    labellist = []
    scorelist = []
    for label, score in zip(_labels,_scores):
        _score_margin = (score[0] - score[1])
        if selection == 'additive':
            if label[0] not in labellist:
                labellist.append(label[0].astype(int))
                scorelist.append(_score_margin)
            else:
                idx = labellist.index(label[0])
                scorelist[idx] += _score_margin
        if selection == 'max':
            if label[0] not in labellist:
                labellist.append(label[0].astype(int))
                scorelist.append(_score_margin)

    # find best prediction
    idx = np.argmax(np.array(scorelist))
    best = (labellist[idx], scorelist[idx])
    return best


if __name__ == "__main__":
    
    with open('edge_test/gpu_win_pc_offload.json', 'r') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df = json_normalize(data,)


    post_prediction = pd.DataFrame()
    for delay_threshold in np.arange(50, 301, 5):
        n = 0
        correct = 0
        conventional = 0
        maximum = 0
        addition = 0
        addition_w = 0
        incorrect = 0
        missed = 0
        sm_additive = 0
        sm_additive_w = 0
        sm_max = 0
        which_exits = np.zeros(4)
        for i, data in df.groupby(['sample']):
            # find predictions within time fram
            exits = len([time for time in data['overall time'].tolist() if time < delay_threshold]) 
            
            if exits != 0:
                n += 1
                which_exits[exits-1] += 1

                # filter predictions within time frame
                labels, scores = np.array(data.prediction.tolist()[:exits]), np.array(data.scores.tolist()[:exits])

                score_additive_w = ScoreMargin(labels, scores, 'additive', weights=[1, 1.5, 2, 2])
                score_additive = ScoreMargin(labels, scores, 'additive', weights=[1,1,1,1])
                score_max = ScoreMargin(labels, scores, 'max')

                labels, scores = MultiHotEncode(labels,scores)
                addtest = Confidence(labels, scores, selection='additive')
                addtest_w = Confidence(labels, scores, selection='additive', weights=[0.6, 1, 2, 2.2])
                
                
                maxtest = Confidence(labels, scores, selection='max')
                target = data.target.tolist()

                addition_w +=  (addtest_w[0]== target[0])
                addition += (addtest[0]==target[0])
                maximum += (maxtest[0] == target[0])
                conventional += (target[0] == data.prediction.tolist()[exits-1][0])
                sm_additive_w += (target[0] == score_additive_w[0])
                sm_additive += (target[0] == score_additive[0])
                sm_max += (target[0]==score_max[0])
            else:
                missed +=1

        if n != 0:
            post_prediction = post_prediction.append({
                'Delay Threshold': delay_threshold,
                'Exit' : which_exits,
                'N': n+missed,
                'missed': missed,
                'latest': conventional / n,
                'confidence (max)' : maximum / n,
                'confidence (add)' : addition / n,
                'confidence (add,weighted)' : addition_w / n,
                'score-margin (max)' : sm_max / n,
                'score_margin (add)' : sm_additive / n,
                'score-margin (add,weighted)' : sm_additive_w / n
            }, ignore_index = True)
        else:
            post_prediction = post_prediction.append({
                'Delay Threshold': delay_threshold,
                'N': n+missed
            }, ignore_index = True)

    print(post_prediction)
    post_prediction.to_json('analysis.json')