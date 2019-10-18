import argparse
import torch
from torch import topk
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

#import branchymodels.BResNet
import datasets

from torch.autograd import Variable
import torch.nn.functional as F

class threshold_tester():
    def __init__(self):
        self.cols = ['threshold', 'test', 'exit', 'sample', 'exited', 'prediction', 'target', 'correct', 'score', 'time']
        self.df = pd.DataFrame(columns=self.cols)
    
    def save(self, name):
        self.df.to_csv(os.path.join('logging', 'threshold_test', name + '.csv'))

    def reset_log(self):
        self.df = pd.DataFrame(columns=self.cols)
    
    def log(self, name, threshold, test, n_exit, sample, exited, prediction, target, correct, score, time):
        self.df = self.df.append(dict(zip(self.cols,[
             threshold, 
             test, 
             n_exit, 
             sample, 
             exited,
             prediction,
             target, 
             correct, 
             score, 
             time])), ignore_index = True)
        
    def confidence_threshold(self, name, threshold_range, model, test_loader, device):
        for sample, (data, target) in enumerate(tqdm(test_loader, leave=False, unit='batch', desc='Testing confidence threshold')):
            if device.type == 'cuda':
                data, target = data.cuda(), target.cuda()
            predictions, timings = model(data)
            for test, threshold in enumerate(threshold_range):
                for n_exit, (pred, time) in enumerate(zip(predictions, timings)):

                    score = F.softmax(pred, dim=1)
                    probability, label = topk(score, k=5)
                    correct = (label.view(-1)[0].item() == target.view(-1).item())

                    if torch.max(probability).item() > threshold: # if model confidence it higher than threshold value
                        exited = 1
                    else:
                        exited = 0
                    self.log(name, threshold, test, n_exit, sample, exited, label.view(-1)[0].item(),  target.view(-1).item(), correct, score.view(-1)[0].item(), time)
            self.save(name)
            
    def score_margin_threshold(self, name, threshold_range, model, test_loader, device):
        for sample, (data, target) in enumerate(tqdm(test_loader, leave=False, unit='batch', desc='Testing score margin threshold = {}')):
            if device.type == 'cuda':
                data, target = data.cuda(), target.cuda()
            predictions, timings = model(data)
            for test, threshold in enumerate(threshold_range):
            

                for n_exit, (pred, time) in enumerate(zip(predictions, timings)):

                    score = F.softmax(pred, dim=1)
                    # I should try to do like in train script if I keep getting too good results...
                    probability, label = topk(score, k=5)
                    correct = (label.view(-1)[0].item() == target.view(-1).item())
                    
                    score_margin = (probability[0][0] - probability[0][1]).item()

                    if score_margin > threshold: # if model confidence it higher than threshold value
                        exited = 1
                    else:
                        exited = 0
                    self.log(name, threshold, test, n_exit, sample, exited, label.view(-1)[0].item(),  target.view(-1).item(), correct, score.view(-1)[0].item(), time)
            self.save(name)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--name', default='branchy', help='run name')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--dataset', default='miniimagenet', help='dataset name')
    parser.add_argument('--n-classes', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_path', default='models/densenet/miniimagenet_10_20191018-092816_model.pth',
                        help='output directory')
    args = parser.parse_args()

    # use cuda if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # seed???
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        model = torch.load(args.model_path)
    else:
        model = torch.load(args.model_path, map_location=torch.device('cpu'))

    _, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.n_classes, device)
    x, _ = test_loader.__iter__().next()

    # load on GPU
    model = model.to(device)

    model.eval()

    tester = threshold_tester()

    thresholds = np.arange(0.1, 1, 0.1)
    with torch.no_grad():
        tester.confidence_threshold(args.name + '_confidence', thresholds, model, test_loader, device)
        tester.reset_log()
        tester.score_margin_threshold(args.name + '_score_margin', thresholds, model, test_loader, device)
        
        
    