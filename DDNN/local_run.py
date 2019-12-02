import argparse
import math
import torch
from torchvision import transforms
from datasets import datasets
import io
from tqdm import tqdm
import numpy as np
import json
from time import perf_counter
import pandas as pd
from dnn.dnn_runner import predictor
import torch.nn.functional as F


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.n_classes, device)

    if device.type == 'cuda':
        model = torch.load(args.model_path)
    else:
        model = torch.load(args.model_path, map_location=torch.device('cpu'))

    results = []

    model.eval()
    with torch.no_grad():
        for sample, (data, target) in enumerate(tqdm(test_loader, leave=False, unit='batch')):
            if device.type == 'cuda':
                data, target = data.cuda(), target.cuda()
            time_start = perf_counter()
            myPredictor = predictor(model, data, args.model_type)
            for ex in range(len(myPredictor.exits)):
                myPredictor.counter = ex
                time_start = perf_counter()
                pred = next(myPredictor)
                pred = pred.cpu()
                score = F.softmax(pred, dim=1)
                prob, pred = torch.topk(score, k=5)
                prob, pred = prob.numpy()[0].tolist(), pred.numpy()[0].tolist()
                prediction_time = (perf_counter() - time_start)*1000 
                msg = {
                    'exit': ex,
                    'prediction': pred,
                    'confidence': prob,
                    'target' : target.view(-1).item(),
                    'time' : prediction_time,
                    'correct' : (pred[0]==target.view(-1).item()),
                    'sample' : sample
                }
                results.append(msg)
    with open('local/' + args.name +'.json', 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Intelligence End-device')
    parser.add_argument('--name', default='run_name')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--dataset', default='miniimagenet-test-only', help='dataset name')
    parser.add_argument('--n-classes', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--model_path', default='models/msdnet/miniimagenet_100_20191029-131509_model.pth',
                        help='output directory')
    parser.add_argument('--model-type', default='msdnet')
    args = parser.parse_args()
    main(args)
