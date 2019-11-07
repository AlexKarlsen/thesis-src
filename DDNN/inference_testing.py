import argparse
import torch
from torch import topk
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F

from time import perf_counter

#import branchymodels.BResNet
from datasets import datasets


class inference_test():
    def __init__(self):
        self.cols = ['threshold', 'exit', 'prediction', 'target', 'correct', 'score_margin', 'time']
        self.df = pd.DataFrame(columns=self.cols)
    
    def save(self, name):
        self.df.to_csv(os.path.join('logging', 'threshold_test', name + '.csv'))
    
    def log(self, threshold, n_exit, prediction, target, correct, score, time):
        self.df = self.df.append(dict(zip(self.cols,[
            threshold,
             n_exit, 
             prediction,
             target, 
             correct, 
             score, 
             time])), ignore_index = True)

    def run_test(self, model_type, model, threshold, data, target):
        if model_type == "early_exit_resnet":
            return self.early_exiting_resnet(model, threshold, data, target)
        elif model_type == "early_exit_densenet":
            return self.early_exiting_densenet(model, threshold, data, target)
        elif model_type == 'msdnet':
            return self.early_exiting_msdnet(model, threshold, data, target)
        else:
            return self.normal_inference(model, threshold, data, target)

    def early_exiting_resnet(self, model, threshold, data, target):   
        time_start = perf_counter()
        model.eval()
        with torch.no_grad():

            ### Exit0 ###
            data = model.conv1(data)
            prediction, data = model.exit1(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            if score_margin > threshold:
                prediction = prediction.data.max(1, keepdim=True)[1]
                return 0, prediction.view(-1).item(), score_margin, perf_counter() - time_start

            ### Exit1 ###
            prediction, data = model.exit2(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            if score_margin > threshold:
                prediction = prediction.data.max(1, keepdim=True)[1]
                return 1, prediction.view(-1).item(), score_margin, perf_counter() - time_start

            ### Exit3 ###
            prediction, data = model.exit3(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            if score_margin > threshold:
                prediction = prediction.data.max(1, keepdim=True)[1]
                return 2, prediction.view(-1).item(), score_margin, perf_counter() - time_start

            ### Exit4 ###
            prediction, _ = model.exit4(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            prediction = prediction.data.max(1, keepdim=True)[1]
            return 3, prediction.view(-1).item(), score_margin, perf_counter() - time_start


    def early_exiting_msdnet(self, model, threshold, data, target):
        time_start = perf_counter()
        model.eval()
        with torch.no_grad():
            for i in range(model.nBlocks-1):
                data = model.blocks[i](data)
                prediction = model.classifier[i](data)
                score = F.softmax(prediction, dim=1)
                probability, label = topk(score, k=2)
                score_margin = self.score_margin(probability)
                if score_margin > threshold:
                    prediction = prediction.data.max(1, keepdim=True)[1]
                    return i, prediction.view(-1).item(), score_margin, perf_counter() - time_start

            # if end exit must be used
            data = model.blocks[model.nBlocks-1](data)
            prediction = model.classifier[model.nBlocks-1](data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            prediction = prediction.data.max(1, keepdim=True)[1]
            return model.nBlocks-1, prediction.view(-1).item(), score_margin, perf_counter() - time_start


    def early_exiting_densenet(self, model, threshold, data, target):   
        time_start = perf_counter()
        model.eval()
        with torch.no_grad():

            ### Exit0 ###
            prediction, data = model.exit1(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            if score_margin > threshold:
                prediction = prediction.data.max(1, keepdim=True)[1]
                return 0, prediction.view(-1).item(), score_margin, perf_counter() - time_start
            data = model.transistion1(data)

            ### Exit1 ###
            prediction, data = model.exit2(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            if score_margin > threshold:
                prediction = prediction.data.max(1, keepdim=True)[1]
                return 1, prediction.view(-1).item(), score_margin, perf_counter() - time_start
            data = model.transistion2(data)

            ### Exit 2 ###
            prediction, data = model.exit3(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            if score_margin > threshold:
                prediction = prediction.data.max(1, keepdim=True)[1]
                return 2, prediction.view(-1).item(), score_margin, perf_counter() - time_start
            data = model.transistion3(data)

            ### Exit 3 ###
            prediction = model.exit4(data)
            score = F.softmax(prediction, dim=1)
            prediction = prediction.data.max(1, keepdim=True)[1]
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            return 3, prediction.view(-1).item(), score_margin, perf_counter() - time_start

    def normal_inference(self, model, threshold, data, target):
        time_start = perf_counter()
        model.eval()
        with torch.no_grad():
            prediction, _ = model(data)
            score = F.softmax(prediction, dim=1)
            prediction = prediction.data.max(1, keepdim=True)[1]
            probability, label = topk(score, k=2)
            score_margin = self.score_margin(probability)
            return 'conventional inference', prediction.view(-1).item(), score_margin, perf_counter() - time_start

    def score_margin(self, score):
        score_margin = (score[0][0] - score[0][1]).item()
        return score_margin
       
    def run(self, name,  thresholds, model_type, model, test_loader):
        for threshold in thresholds:
            for (data, target) in tqdm(test_loader, leave=False, unit='batch'):
                data, target = data.cuda(), target.cuda()
                n_exit, prediction, score, time =  self.run_test(model_type, model, threshold, data, target)
                correct = (prediction == target).view(-1).item()
                self.log(threshold, n_exit, prediction, target.view(-1).item(), correct, score, time)
        self.save(name)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--name', default='densenet-after-del', help='run name')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--dataset', default='miniimagenet', help='dataset name')
    parser.add_argument('--n-classes', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_path', default='models/densenet/miniimagenet_100_20191018-165914_model.pth',
                        help='output directory')
    parser.add_argument('--model-type', default='early_exit_densenet', help='run name')
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


    tester = inference_test()

    thresholds = np.linspace(0.9, 1, 10)
    #thresholds =[0.5]
    
    tester.run(args.name + '_inference_test', thresholds, args.model_type, model, test_loader)
    
