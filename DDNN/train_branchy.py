from __future__ import print_function

import argparse
import os
from tqdm import tqdm

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

from time import perf_counter, strftime # uses the most precise clock on OS 
import copy
import numpy as np
import pandas as pd

from logger import Logger

import datasets
from BResNet import BResNet as net

def train(model, branch_weights, train_loader, optimizer):

    # setting to train mode. This could be refactored see transfer learning tutorial
    model.train()

    model_losses = np.zeros(model.branches)
    num_correct = np.zeros(model.branches)
    timings_arr = []

    # start timing training
    time_start = perf_counter()
        
    # iterate data
    for data, target in tqdm(train_loader, leave=False, unit='batch', desc='Training'):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)

        #zero the parameter gradients
        optimizer.zero_grad()

        # run model on input
        predictions, timings = model(data)
        total_loss = 0
        # for each branch prediction, add to loss
        for i, (prediction, weight) in enumerate(zip(predictions, branch_weights)):
            # compute the loss
            loss = F.cross_entropy(prediction, target)
            # add loss to list
            model_losses[i] += loss.sum()*len(target)
            total_loss += (loss * weight)
            #what exactly happens here?
            pred = prediction.data.max(1, keepdim=True)[1]
            correct = (pred.view(-1) == target.view(-1)).long().sum().item()
            num_correct[i] += correct
            timings_arr.append(timings)

        # backpropagate
        total_loss.backward()
        optimizer.step()

    N = len(train_loader.dataset)

    time_run = perf_counter() - time_start

    # return losses and scores for visualization
    model_losses = [i.item() / N for i in model_losses]
    scores = [i / N for i in num_correct]
    print('-' * 90)
    loss_str = ', '.join(['branch-{}: {:.4f}'.format(i, loss)
                        for i, loss in enumerate(model_losses)])
    acc_str = ', '.join(['branch-{}: {:.4f}'.format(i, correct)
                        for i, correct in enumerate(scores)])
    print('Train Time: {:.0f}s'.format(time_run))                        
    print('Train Loss: [{}]'.format(loss_str))
    print('Train Acc.: [{}]'.format(acc_str))
    print('-' * 90)

    return model_losses, scores, np.mean(timings_arr, axis=0)

def test(model, test_loader):
    model.eval()
    model_losses = np.zeros(model.branches)
    num_correct = np.zeros(model.branches)
    timings_arr = np.empty((0,4))
    # start timing inference
    time_start = perf_counter()

    # do not compute gradient for testing. 
    # if not we run out of memory
    with torch.no_grad():
        # iterate test data
        for data, target in tqdm(test_loader, leave=False, unit='batch', desc='Testing'):
            data, target = data.cuda(), target.cuda()
            predictions, timings = model(data)

            # for all predictions
            for i, prediction in enumerate(predictions):
                # determine loss
                loss = F.cross_entropy(prediction, target)
                # get the most certain prediction
                pred = prediction.data.max(1, keepdim=True)[1]
                # correct prediction or not?
                correct = (pred.view(-1) == target.view(-1)).long().sum().item()
                # add to correct counter
                num_correct[i] += correct
                # add to loss counter
                model_losses[i] += loss.sum()*len(target)
                timings_arr = np.append(timings_arr, np.array([timings]),axis=0)

    # end timing training
    time_run = perf_counter() - time_start
    

    N = len(test_loader.dataset)
    # return losses and scores for visualization
    model_losses = [i.item() / N for i in model_losses]
    scores = [i / N for i in num_correct]

    loss_str = ', '.join(['branch-{}: {:.4f}'.format(i, loss)
                        for i, loss in enumerate(model_losses)])
    acc_str = ', '.join(['branch-{}: {:.4f}'.format(i, score)
                        for i, score in enumerate(scores)])
    print('Test time: {:.0f}s'.format(time_run))
    print('Test Loss: [{}]'.format(loss_str))
    print('Test Acc.: [{}]'.format(acc_str))
    print('-' * 90)

    return model_losses, scores, np.mean(timings_arr,axis=0)

def train_model(model, name, model_path, train_loader, test_loader, lr, epochs, branch_weights):
    # Data logging
    logger = Logger(name, model.branches)

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = np.zeros(model.branches)

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)

    # if model directory does not exits, then create it
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
 
    # train test lop
    for epoch in range(1, epochs+1):
        print('[Epoch {}/{}]'.format(epoch,epochs))

        # Run train and test and get data
        train_loss, train_acc, train_time = train(model, branch_weights, train_loader, optimizer)
        test_loss, test_acc, test_time = test(model, test_loader)

        # Save best model i.e. early stopping
        if np.mean(test_acc) > np.mean(best_acc):
            best_acc = test_acc
            # best_model_wts = copy.deepcopy(model.state_dict())
            print('Saving new best model')
            torch.save(model, model_path)

        # log epoch stats
        logger.log(train_loss, train_acc, train_time, test_loss, test_acc, test_time)

        scheduler.step()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='BranchyNet training script')
    parser.add_argument('--name', default='branchy', help='run name')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--dataset', default='imagenet', help='dataset name')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--n-classes', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')  
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--continue_training', action='store_true',
                        help='continue training from checkpoint')
    parser.add_argument("--branch-weights", nargs=4, metavar=('a', 'b', 'c', 'd'),
                         help="my help message", type=float, default=[1.0, 1.0, 1.0, 1.0])
    args = parser.parse_args()

    timestr = strftime("%Y%m%d-%H%M%S")
    name = args.name + '_' + timestr
    # use cuda if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # seed???
    torch.manual_seed(args.seed)
    if device == 'cuda:0':
        torch.cuda.manual_seed(args.seed)

    # get input data
    train_loader, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.n_classes, device)

    # get channels from input data
    x, _ = train_loader.__iter__().next()
    in_channels = x.shape[2]
    out_channels = args.n_classes

    # construct DDNN
    model = net(args.n_classes)

    model_path = os.path.join('models', args.name, args.dataset + '_' + str(args.n_classes) + '_' + timestr + '_model.pth')
    if args.continue_training:
        model = torch.load(model_path)

    # load on GPU
    model = model.to(device)

    # Run training
    print()
    print('Training starting: [dataset: {}, number of epochs: {}, batch-size: {}, initial learning rate: {}]'.format(
        args.dataset,
        args.epochs,
        args.batch_size,
        args.lr
    ))
    print('-' * 90)
    time_since = perf_counter()
    train_model(model, name, model_path, train_loader, test_loader, args.lr, args.epochs, args.branch_weights)

    time_elapsed = perf_counter() - time_since
    minutes, seconds = divmod(time_elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    print('Training completed in {:.0f}h{:.0f}m{:.2f}s'.format(hours, minutes, seconds))
    
    