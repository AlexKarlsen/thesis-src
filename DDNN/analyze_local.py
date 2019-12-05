import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import argparse
from tqdm import tqdm

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
    return np.array(retLabels), np.array(retScores)

def Confidence(_labels, _scores, n_classes, selection='additive', weights = None):
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
    if n_classes != _labels.shape[0]:
            _labels, _scores = MultiHotEncode(_labels, _scores, n_classes)
    
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
    ##### should i sort??? i font think so...
    # find best prediction
    idx = np.argmax(np.array(scorelist))
    best = (labellist[idx], scorelist[idx])
    return best

def delay_threshold_test(df, n_classes, platform, model_type):
    post_prediction = pd.DataFrame()
    for delay_threshold in tqdm(np.arange(3, 270,1)):
        n = conventional = maximum = addition = addition_w = missed = sm_additive = sm_additive_w = sm_max = 0
        if model_type == 'msdnet':
            n_exits = 5
        elif model_type == 'b-resnet' or model_type == 'b-densenet':
            n_exits = 4
        else:
            n_exits = 1
        which_exits = np.zeros(n_exits)
        for i, data in df.groupby(['sample']):
            # find predictions within time frame

            timings = data['time'].tolist()
            cum_timings = np.cumsum(timings)
            exits = len([time for time in cum_timings if time < delay_threshold])
            
            n += 1
            if exits != 0:
                
                which_exits[exits-1] += 1

                # filter predictions within time frame
                #labels, scores = np.array(data.prediction.tolist()[:exits]), np.array(data.scores.tolist()[:exits])

                #score_additive_w = ScoreMargin(labels, scores, 'additive', args.weights)
                #score_additive = ScoreMargin(labels, scores, 'additive')
                #score_max = ScoreMargin(labels, scores, 'max')

                #labels, scores = MultiHotEncode(labels,scores)
                #addtest = Confidence(labels, scores, n_classes, selection='additive')
                #addtest_w = Confidence(labels, scores, n_classes, selection='additive', weights=args.weights)
                
                
                #maxtest = Confidence(labels, scores, n_classes, selection='max')
                target = data.target.tolist()

                #addition_w +=  (addtest_w[0]== target[0])
                #addition += (addtest[0]==target[0])
                #maximum += (maxtest[0] == target[0])
                conventional += (target[0] == data.prediction.tolist()[exits-1][0])
                #sm_additive_w += (target[0] == score_additive_w[0])
                #sm_additive += (target[0] == score_additive[0])
                #sm_max += (target[0]==score_max[0])
            else:
                missed +=1

        if n != 0:
            post_prediction = post_prediction.append({
                'Delay Threshold': delay_threshold,
                'Exit' : which_exits,
                'N': n,
                'missed': missed,
                'latest': conventional / n,
                #'confidence (max)' : maximum / n,
                #'confidence (add)' : addition / n,
                #'confidence (add,weighted)' : addition_w / n,
                #'score-margin (max)' : sm_max / n,
                #'score_margin (add)' : sm_additive / n,
                #'score-margin (add,weighted)' : sm_additive_w / n
            }, ignore_index = True)
        else:
            post_prediction = post_prediction.append({
                'Delay Threshold': delay_threshold,
                'N': n+missed
            }, ignore_index = True)

    print(post_prediction)
    post_prediction.to_json('local/' + platform + '_local_' + model_type + '_analysis.json')

def lost_prediction_test(df, args):
    
    post_prediction = []
    if args.model_type == 'msdnet':
        exits = 5
    else:
        exits = 4
    for k in range(1,exits+1):

        n = conventional = maximum = addition = addition_w = sm_additive = sm_additive_w = sm_max = 0
        for _, data in df.groupby(['sample']):
            n += 1
            labels, scores = np.array(data.prediction.tolist()[:k]), np.array(data.scores.tolist()[:k])

            score_additive_w = ScoreMargin(labels, scores, 'additive', weights=args.weights)
            score_additive = ScoreMargin(labels, scores, 'additive')
            score_max = ScoreMargin(labels, scores, 'max')

            labels, scores = MultiHotEncode(labels,scores)
            addtest = Confidence(labels, scores, selection='additive')
            addtest_w = Confidence(labels, scores, selection='additive', weights=args.weights)
            
            
            maxtest = Confidence(labels, scores, selection='max')
            target = data.target.tolist()

            addition_w +=  (addtest_w[0]== target[0])
            addition += (addtest[0]==target[0])
            maximum += (maxtest[0] == target[0])
            conventional += (target[0] == data.prediction.tolist()[k-1][0])
            sm_additive_w += (target[0] == score_additive_w[0])
            sm_additive += (target[0] == score_additive[0])
            sm_max += (target[0]==score_max[0])

        post_prediction.append({
            'exit' : k-1,
            'N Exits' : n,
            'latest': conventional / n,
            'confidence (max)' : maximum / n,
            'confidence (add)' : addition / n,
            'confidence (add,weighted)' : addition_w / n,
            'score-margin (max)' : sm_max / n,
            'score_margin (add)' : sm_additive / n,
            'score-margin (add,weighted)' : sm_additive_w / n
        })

    post_prediction = pd.DataFrame(post_prediction)
    post_prediction.to_json(args.name +'_lost_prediction_analysis.json')

def main(platform, model_type, test):
    with open('local/' + platform + '_local_' + model_type + '.json', 'r') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df = json_normalize(data,)

    if test == 'delay-threshold':
        delay_threshold_test(df,  100, platform, model_type)
    elif args.test == 'lost-prediction':
        lost_prediction_test(df, args)
    else:
        raise Exception('test must be specified')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze edge offloading results')
    parser.add_argument('--platform', default='gpu')
    parser.add_argument('--test', default='delay-threshold'),
    parser.add_argument('--model-type', default='b-resnet'),
    parser.add_argument('--weights', default=[1,1.2,1.4,1.6,1.6])
    args = parser.parse_args()
    with open('local/' + args.platform + '_local_' + args.model_type + '.json', 'r') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame()
    df = json_normalize(data,)

    if args.test == 'delay-threshold':
        delay_threshold_test(df,  100, args.platform, args.model_type)
    elif args.test == 'lost-prediction':
        lost_prediction_test(df, args)
    else:
        raise Exception('test must be specified')
    