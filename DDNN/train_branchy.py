from __future__ import print_function

import argparse
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

import time
import copy
import pandas as pd
from statistics import mean

import datasets
from BResNet import BResNet as net

def train(model, train_loader, optimizer):

    # setting to train mode. This could be refactored see transfer learning tutorial
    model.train()

    model_losses = [0]*(model.branches)
    num_correct = [0]*(model.branches)

    # start timing training
    time_start = time.time()
        
    # iterate data
    for data, target in tqdm(train_loader, leave=False, unit='batch', desc='Training'):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)

        # The loss for the entire DDNN
        total_loss = 0

        #zero the parameter gradients
        optimizer.zero_grad()

        # run model on input
        predictions = model(data)
        
        # for each branch prediction, add to loss
        for i, prediction in enumerate(predictions):
            # compute the loss
            loss = F.cross_entropy(prediction, target)
            # add loss to list
            model_losses[i] += loss.sum()*len(target)
            # the loss for entire DDNN
            total_loss += loss
            #what exactly happens here?
            pred = prediction.data.max(1, keepdim=True)[1]
            correct = (pred.view(-1) == target.view(-1)).long().sum().item()
            num_correct[i] += correct
            
        # backpropagate
        total_loss.backward()
        optimizer.step()

    N = len(train_loader.dataset)

    time_run = time.time() - time_start

    # return losses and scores for visualization
    model_losses = [i.item() / N for i in model_losses]
    scores = [i / N for i in num_correct]
    print('-' * 90)
    loss_str = ', '.join(['branch-{}: {:.4f}'.format(i, loss)
                        for i, loss in enumerate(model_losses)])
    acc_str = ', '.join(['branch-{}: {:.4f}'.format(i, correct)
                        for i, correct in enumerate(scores)])
    print('Train Time: {:0f}s'.format(time_run))                        
    print('Train Loss: [{}]'.format(loss_str))
    print('Train Acc.: [{}]'.format(acc_str))
    print('-' * 90)

    #train_data = { 'Edge train loss':  model_losses[-1]}

    
    return model_losses, scores, time_run

def test(model, test_loader, branches=4):
    model.eval()
    model_losses = [0]*(branches)
    num_correct = [0]*(branches)

    # start timing inference
    time_start = time.time()

    # do not compute gradient for testing. 
    # if not we run out of memory
    with torch.no_grad():
        # iterate test data
        for data, target in tqdm(test_loader, leave=False, unit='batch', desc='Testing'):
            data, target = data.cuda(), target.cuda()
            predictions = model(data)

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

    # end timing training
    time_run = time.time() - time_start

    N = len(test_loader.dataset)
    # return losses and scores for visualization
    model_losses = [i.item() / N for i in model_losses]
    scores = [i / N for i in num_correct]

    loss_str = ', '.join(['branch-{}: {:.4f}'.format(i, loss)
                        for i, loss in enumerate(model_losses)])
    acc_str = ', '.join(['branch-{}: {:.4f}'.format(i, score)
                        for i, score in enumerate(scores)])
    print('Test time: {:0f}s'.format(time_run))
    print('Test Loss: [{}]'.format(loss_str))
    print('Test Acc.: [{}]'.format(acc_str))
    print('-' * 90)

    
    return model_losses, scores, time_run

def train_model(model, model_path, train_loader, test_loader, lr, epochs, rough_tune, name, unfreeze_base):
    # for params in model.exit1branch.parameters():
    #     params.requires_grad = False
    # for params in model.exit2branch.parameters():
    #     params.requires_grad = False
    #ps = filter(lambda x: x.requires_grad, model.parameters())

    # Data logging
    cols = []
    for v in ['train', 'test']:
        for u in ['loss', 'accuracy', 'time']:
            cols +=  ['branch-'+ str(index)+ '-' + v + '-' + u  for index in range(4)]
    df = pd.DataFrame(columns=cols)

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    # rough-tuning
    # if rough_tune != 0:
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=rough_tune, gamma=0.1)

    # if model directory does not exits, then create it
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
 
    # train test lop
    for epoch in range(1, epochs+1):
        print('[Epoch {}/{}]'.format(epoch,epochs))

        # # freezing base to rough tune classifier
        # if rough_tune:
        #     for i in model.model.parameters():
        #         i.requires_grad = False

        # # start fine-tuning
        # if epoch == rough_tune + 1:
        #     lr = lr * 0.01 # should this be a parameter
        #     print('Switching to cosine annealing scheduler with much lower learning rate, lr={}'.format(lr))
        #     optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs)

        if unfreeze_base: 
            for i in model.model.parameters():
                i.requires_grad = True

        # Run train and test and get data
        train_loss, train_acc, train_time = train(model, train_loader, optimizer)
        test_loss, test_acc, test_time = test(model, test_loader)

        # deep copy the model
        if mean(test_acc) > mean(best_acc):
            best_acc = test_acc
            # best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, model_path + 'model.pth')
        data = []
        # format training and testing data
        data.append(train_loss)
        data.append(train_acc)
        data.append(train_time)
        data.append(test_loss)
        data.append(test_acc)
        data.append(test_time)
        data_dict = dict(zip(cols, data))

        # save the models for inference
        torch.save(model, model_path + name + '.pth')

        scheduler.step()

        # continously log results
        df = df.append(data_dict, ignore_index=True)
        df.to_csv('logging/train_data_' + name + '.csv')

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Example')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--dataset', default='voc', help='dataset name')
    parser.add_argument('--name', default='ddnn', help='run name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output', default='models/ddnn/',
                        help='output directory')
    parser.add_argument('--continue_training', action='store_true',
                        help='continue training from checkpoint')
    parser.add_argument('--rough_tune', type=int, default=5,
                        help='rough tuning for n epochs')
    parser.add_argument('--unfreeze_base', action='store_true', help='unfreeze base')

    args = parser.parse_args()

    # use cuda if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # seed???
    torch.manual_seed(args.seed)
    if device == 'cuda:0':
        torch.cuda.manual_seed(args.seed)

    # get input data
    train_dataset, train_loader, test_dataset, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, device)

    # get channels from input data
    x, _ = train_loader.__iter__().next()
    in_channels = x.shape[2]
    out_channels = len(train_dataset.classes)

    
    # construct DDNN
    model = net(out_channels)

    if args.continue_training:
        model = torch.load(args.output + 'model.pth')

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
    time_since = time.time()
    train_model(model, args.output, train_loader, test_loader, args.lr, args.epochs, args.rough_tune, args.name, args.unfreeze_base)

    
    print('Training completed in {}'.format(time.time()-time_since))
    
    