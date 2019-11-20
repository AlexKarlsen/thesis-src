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
                pred = next(myPredictor)
                score = F.softmax(pred, dim=1).max(1)[0].item()
                pred = pred.data.max(1, keepdim=True)[1].item()
                msg = {
                    'exit': ex,
                    'prediction': pred,
                    'confidence': score,
                    'target' : target.view(-1).item(),
                    'time' : (perf_counter() - time_start) * 1000,
                    'correct' : (pred==target.view(-1).item()),
                    'sample' : sample
                }
                results.append(msg)
    log = pd.DataFrame(results)
    log.to_csv(args.name + '.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Intelligence End-device')
    parser.add_argument('--name', default='run_name')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--dataset', default='miniimagenet-test-only', help='dataset name')
    parser.add_argument('--n-classes', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--model_path', default='models/resnet101/miniimagenet_100_20191025-161132_model.pth',
                        help='output directory')
    parser.add_argument('--model-type', default='resnet')
    args = parser.parse_args()
    main(args)