import argparse
import torch
from torch import topk
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from time import perf_counter

#import branchymodels.BResNet
import datasets


class inference_test():
    def __init__(self):
        self.cols = ['threshold', 'test', 'exit', 'sample', 'exited', 'prediction', 'target', 'correct', 'score', 'time']
        self.df = pd.DataFrame(columns=self.cols)
    
    def save(self, name):
        self.df.to_csv(os.path.join('logging', 'threshold_test', name + '.csv'))
    
    def log(self, name, n_exit, prediction, target, correct, score, time):
        self.df = self.df.append(dict(zip(self.cols,[
             n_exit, 
             prediction,
             target, 
             correct, 
             score, 
             time])), ignore_index = True)

    def early_exiting(self, model, data, target):   
        time_start = perf_counter()

        prediction, _ = model.exit1(data)
        score = F.softmax(prediction)
        if self.score_margin(score) > threshold:
            prediction = prediction.data.max(1, keepdim=True)[1]
            return 0, prediction.view(-1).item(), target.view(-1).item(), score[0][0].item(), time_start - perf_counter()
        prediction = model.exit2(data)
        score = F.softmax(prediction)
        if self.score_margin(F.softmax(prediction)) > threshold:
            prediction = prediction.data.max(1, keepdim=True)[1]
            return 1, prediction.view(-1).item(), target.view(-1).item(), score[0][0.item()], time_start - perf_counter()
        prediction = model.exit3(data)
        score = F.softmax(prediction)
        if self.score_margin(F.softmax(prediction)) > threshold:
            prediction = prediction.data.max(1, keepdim=True)[1]
            return 2, prediction.view(-1).item(), target.view(-1).item(), score[0][0].item(), time_start - perf_counter()
        prediction = model.exit4(data) # oops densenet must implement same structure...
        score = F.softmax(prediction)
        if self.score_margin(F.softmax(prediction)) > threshold:
            prediction = prediction.data.max(1, keepdim=True)[1]
            return 3, prediction.view(-1).item(), target.view(-1).item(), score[0][0].item(), time_start - perf_counter()

    def score_margin(self, score):
        score_margin = (score[0][1].item() - score[0][1].item())
        return score_margin
       
    def run(self, model):
        for sample, (data, target) in enumerate(tqdm(test_loader, leave=False, unit='batch', desc='Testing confidence threshold = {}'.format(threshold))):
            data, target = data.cuda(), target.cuda()
            n_exit, prediction, target, score, time =  self.early_exiting(model, data, target)
            correct = (prediction == target)
            self.log(name, n_exit, prediction, target, correct, score, time)
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
    parser.add_argument('--model_path', default='models/dense/miniimagenet_10_20191001-121910_model.pth',
                        help='output directory')
    args = parser.parse_args()

    # use cuda if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # seed???
    torch.manual_seed(args.seed)
    if device == 'cuda:0':
        torch.cuda.manual_seed(args.seed)

    _, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.n_classes, device)
    x, _ = test_loader.__iter__().next()

    model = torch.load(args.model_path)
    
    # load on GPU
    model = model.to(device)

    model.eval()

    tester = threshold_tester()

    thresholds = np.arange(0.1, 1, 0.1)
    with torch.no_grad():
        tester.confidence_threshold(args.name + '_confidence', thresholds, model, test_loader)
        tester.score_margin_threshold(args.name + '_score_margin', thresholds, model, test_loader)
    