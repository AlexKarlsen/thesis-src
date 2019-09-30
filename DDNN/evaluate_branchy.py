import argparse
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

import datasets

from torch.autograd import Variable
import torch.nn.functional as F

def confidence_threshold(threshold_range, model, test_loader):
    for test, threshold in enumerate(threshold_range):
        for sample, (data, target) in enumerate(tqdm(test_loader, leave=False, unit='batch', desc='Testing')):
            data, target = data.cuda(), target.cuda()
            predictions, timings = model(data)

            cols = ['test', 'exit', 'sample', 'threshold', 'exited', 'correct', 'confidence', 'time']
            df = pd.DataFrame(columns=cols)
            
            for n_exit, (pred, time) in enumerate(zip(predictions, timings)):

                confidence = F.softmax(pred, dim=1)
                correct = (pred.view(-1) == target.view(-1)).long().sum().item()
                
                if confidence.argmax(dim=1) > threshold: # if model confidence it higher than threshold value
                    exited = 1
                else:
                    exited = 0
                df = df.append([test, n_exit, sample, exited, correct, confidence, time])
    df.to_csv(os.path.join('logging', 'threshold_test', 'name.csv'))
        

        

def score_margin_threshold(threshold_range, test_loader):
    pass

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--dataset', default='miniimagenet', help='dataset name')
    parser.add_argument('--n-classes', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_path', default='models/ddnn/ddnn.pth',
                        help='output directory')
    args = parser.parse_args()

    # use cuda if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # seed???
    torch.manual_seed(args.seed)
    if device == 'cuda:0':
        torch.cuda.manual_seed(args.seed)

    _, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.n_classes, args.cuda)
    x, _ = test_loader.__iter__().next()

    model = torch.load(args.model_path)

    thresholds = np.arange(0.1, 0.9, 0.1)
    confidence_threshold(thresholds, model, test_loader)
    