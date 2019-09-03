from __future__ import print_function

import argparse
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

import time
import pandas as pd

import datasets
from DDNN import DDNN as net

def train(model, train_loader, optimizer, num_devices):

    # setting to train mode. This could be refactored see transfer learning tutorial
    model.train()

    # seems nifty, but what happen exactly here?...
    model_losses = [0]*(num_devices + 1)
    num_correct = [0]*(num_devices + 1)

    # start timing training
    time_start = time.time()
        
    # iterate data
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, leave=False)):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)

        # The loss for the entire DDNN
        total_loss = 0

        #zero the parameter gradients
        optimizer.zero_grad()

        # run model on input
        predictions = model(data)
        
        # for each prediction (num_devices + 1 for edge), add to loss
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

    # return losses and scores for visualization
    model_losses = [i.item() / N for i in model_losses]
    scores = [i / N for i in num_correct]
    print('-' * 30)
    loss_str = ', '.join(['dev-{}: {:.4f}'.format(i, loss)
                        for i, loss in enumerate(model_losses[:-1])])
    acc_str = ', '.join(['dev-{}: {:.4f}'.format(i, correct)
                        for i, correct in enumerate(scores[:-1])])
    print('Train Time: {:0f}s'.format(time.time()-time_start))                        
    print('Train Loss: [{}, edge-{:.4f}]'.format(loss_str, model_losses[-1]))
    print('Train Acc.: [{}, edge-{:.4f}]'.format(acc_str, scores[-1]))
    print('-' * 30)

    #train_data = { 'Edge train loss':  model_losses[-1]}

    
    return model_losses, scores

def test(model, test_loader, num_devices):
    model.eval()
    model_losses = [0]*(num_devices + 1)
    num_correct = [0]*(num_devices + 1)

    # start timing inference
    time_start = time.time()

    # do not compute gradient for testing. 
    # if not we run out of memory
    with torch.no_grad():
        # iterate test data
        for data, target in tqdm(test_loader, leave=False):
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
    time_end = time.time()

    N = len(test_loader.dataset)
    # return losses and scores for visualization
    model_losses = [i.item() / N for i in model_losses]
    scores = [i / N for i in num_correct]

    loss_str = ', '.join(['dev-{}: {:.4f}'.format(i, loss)
                        for i, loss in enumerate(model_losses[:-1])])
    acc_str = ', '.join(['dev-{}: {:.4f}%'.format(i, score)
                        for i, score in enumerate(scores[:-1])])
    print('Test time: {:0f}s'.format(time_end-time_start))
    print('Test Loss: [{}, edge-{:.4f}]'.format(loss_str, model_losses[-1]))
    print('Test Acc.: [{}, edge-{:.4f}]'.format(acc_str, scores[-1]))
    print('-' * 30)

    
    return model_losses, scores

def train_model(model, model_path, train_loader, test_loader, lr, epochs, num_devices, rough_tune):
    #ps = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    # rough-tuning
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=rough_tune, gamma=0.1)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    data_arr = []

    for epoch in range(1, epochs+1):
        print('[Epoch {}/{}]'.format(epoch,epochs))

        # swicth to cosine annealing with much lower lr
        if epoch == rough_tune + 1:
            lr = scheduler.get_lr()
            print('Switching to cosine annealing scheduler with much lower learning rate, lr={}'.format(lr))
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Run train and test and get data
        train_loss, train_acc = train(model, train_loader, optimizer, num_devices)
        test_loss, test_acc = test(model, test_loader, num_devices)

        # format training and testing data
        data = train_loss + train_acc + test_loss + test_acc
        data_dict = dict(zip(cols, data))
        data_arr.append(data_dict)

        # save all the models for inference
        torch.save(model, model_path + 'ddnn.pth')
        torch.save(model.edge_model, model_path + 'edge.pth')
        for i, device in enumerate(model.device_models):
            torch.save(device, model_path + 'dev' + str(i) + '.pth')

        scheduler.step()

    return data_arr

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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output', default='models/ddnn/',
                        help='output directory')
    parser.add_argument('--n_devices', default=5,
                        help='number of devices')
    parser.add_argument('--continue_training', action='store_true',
                        help='continue training from checkpoint')
    parser.add_argument('--rough_tune', default=5,
                        help='rough tuning for n epochs')

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

    # path to saved model 
    model_path = args.output
    
    # construct DDNN
    model = net(in_channels, out_channels, args.n_devices)

    if args.continue_training:
        model = torch.load(model_path + 'ddnn.pth')

    # load on GPU
    model = model.to(device)

    # Data logging
    cols = []
    for v in ['train', 'test']:
        for u in ['loss', 'accuracy']:
            cols +=  ['dev-'+ str(index)+ '-' + v + '-' + u  for index in range(args.n_devices)]
            cols += ['edge-' + v + '-' + u] 
    df = pd.DataFrame(columns=cols)

    # Run training
    print()
    print('Training staring: [dataset: {}, number of epochs: {}, batch-size: {}, initial learning rate: {}, number of devices {}]'.format(
        args.dataset,
        args.epochs,
        args.batch_size,
        args.lr,
        args.n_devices
    ))
    print('-'*30)
    time_since = time.time()
    data = train_model(model, args.output, train_loader, test_loader, args.lr, args.epochs, args.n_devices, args.rough_tune)

    
    print('Training completed in {}'.format(time_since-time.time()))
    
    df = df.append(data, ignore_index=True)
    df.to_csv('logging/train_data_' + str(time.time()) + '.csv')