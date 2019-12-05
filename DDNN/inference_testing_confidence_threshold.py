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
        self.cols = ['threshold', 'exit', 'prediction', 'target', 'correct', 'confidence', 'time']
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

    def run_test(self, model_type, model, threshold, data):
        if model_type == "b-resnet":
            return self.early_exiting_resnet(model, threshold, data)
        elif model_type == "b-densenet":
            return self.early_exiting_densenet(model, threshold, data)
        elif model_type == 'msdnet':
            return self.early_exiting_msdnet(model, threshold, data)
        else:
            return self.normal_inference(model, threshold, data)

    def early_exiting_resnet(self, model, threshold, data):   
        time_start = perf_counter()
        model.eval()
        with torch.no_grad():

            ### Exit0 ###
            data = model.conv1(data)
            prediction, data = model.exit1(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)
            if probability.view(-1).item() > threshold:
                return 0, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start

            ### Exit1 ###
            prediction, data = model.exit2(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)
            if probability.view(-1).item() > threshold:
                return 1, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start

            ### Exit3 ###
            prediction, data = model.exit3(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)
            if probability.view(-1).item() > threshold:
                return 2, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start

            ### Exit4 ###
            prediction, _ = model.exit4(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)
            
            return 3, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start


    def early_exiting_msdnet(self, model, threshold, data):
        time_start = perf_counter()
        model.eval()
        with torch.no_grad():
            for i in range(model.nBlocks-1):
                data = model.blocks[i](data)
                prediction = model.classifier[i](data)
                score = F.softmax(prediction, dim=1)
                probability, label = topk(score, k=1)
                prediction = prediction.data.max(1, keepdim=True)[1]
                if probability.view(-1).item() > threshold:
                    return i, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start

            # if end exit must be used
            data = model.blocks[model.nBlocks-1](data)
            prediction = model.classifier[model.nBlocks-1](data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)

            return 4, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start



    def early_exiting_densenet(self, model, threshold, data):   
        time_start = perf_counter()
        model.eval()
        with torch.no_grad():

            ### Exit0 ###
            prediction, data = model.exit1(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)
            if probability.view(-1).item() > threshold:
                return 0, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start
            data = model.transistion1(data)

            ### Exit1 ###
            prediction, data = model.exit2(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)
            if probability.view(-1).item() > threshold:
                return 1, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start
            data = model.transistion2(data)

            ### Exit 2 ###
            prediction, data = model.exit3(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)
            if probability.view(-1).item() > threshold:
                return 2, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start
            data = model.transistion3(data)

            ### Exit 3 ###
            prediction = model.exit4(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)

            return 3, label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start

    def normal_inference(self, model, threshold, data):
        time_start = perf_counter()
        model.eval()
        with torch.no_grad():
            prediction, _ = model(data)
            score = F.softmax(prediction, dim=1)
            probability, label = topk(score, k=1)
            return 'c', label.view(-1).item(), probability.view(-1).item(), perf_counter() - time_start

    def score_margin(self, score):
        score_margin = (score[0][0] - score[0][1]).item()
        return score_margin
       
    def run(self, name,  thresholds, model_type, model, test_loader, force_cpu=False):
        for threshold in thresholds:
            for (data, target) in tqdm(test_loader, leave=False, unit='batch'):
                if torch.cuda.is_available() and force_cpu is not True:
                    #print('data on cuda')
                    data, target = data.cuda(), target.cuda()
                n_exit, prediction, score, time =  self.run_test(model_type, model, threshold, data)
                correct = (prediction == target.view(-1).item())
                self.log(threshold, n_exit, prediction, target.view(-1).item(), correct, score, time)
        self.save(name)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--name', default='b-resnet_confidence_', help='run name')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--dataset', default='miniimagenet-test-only', help='dataset name')
    parser.add_argument('--n-classes', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_path', default='models/b-resnet/miniimagenet_100_20191023-162944_model.pth',
                        help='output directory')
    parser.add_argument('--model-type', default='b-resnet', help='run name')
    parser.add_argument('--force-cpu', default=False)
    args = parser.parse_args()

    # use cuda if available else use cpu
    if args.force_cpu is not True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    # seed???
    torch.manual_seed(args.seed)
    if device == 'cuda:0':
        torch.cuda.manual_seed(args.seed)

    _, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.n_classes, device)
    #x, _ = test_loader.__iter__().next()

    if device.type == 'cuda' and args.force_cpu is not True:
        torch.cuda.manual_seed(args.seed)
        model = torch.load(args.model_path)
    else:
        model = torch.load(args.model_path, map_location=torch.device('cpu'))

    print(device.type)

    tester = inference_test()

    thresholds = np.arange(0.1,1,0.1)
    #thresholds =[0.5]
    
    tester.run(args.name + '_inference_test', thresholds, args.model_type, model, test_loader, args.force_cpu)
    