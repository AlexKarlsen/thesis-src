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
        
        # for each prediction (num_devices + 1 for cloud), add to loss
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

    # end timing training
    time_end = time.time()

    N = len(train_loader.dataset)
    loss_str = ', '.join(['dev-{}: {:.4f}'.format(i, loss.item() / N)
                        for i, loss in enumerate(model_losses[:-1])])
    acc_str = ', '.join(['dev-{}: {:.4f}%'.format(i, 100. * (correct / N))
                        for i, correct in enumerate(num_correct[:-1])])
    print('Training elapsed in {:0f}s'.format(time_end-time_start))                        
    print('Train Loss:: {}, cloud-{:.4f}'.format(loss_str, model_losses[-1].item() / N))
    print('Train  Acc.:: {}, cloud-{:.4f}%'.format(acc_str, 100. * (num_correct[-1] / N)))

    train_data = { 'Edge train loss':  model_losses[-1].item() / N }

    return model_losses

 

def test(model, test_loader, num_devices):
    model.eval()
    model_losses = [0]*(num_devices + 1)
    num_correct = [0]*(num_devices + 1)

    # start timing inference
    time_start = time.time()

    for data, target in tqdm(test_loader, leave=False):
        data, target = data.cuda(), target.cuda()
        predictions = model(data)
        for i, prediction in enumerate(predictions):
            loss = F.cross_entropy(prediction, target, reduction='sum').item()
            pred = prediction.data.max(1, keepdim=True)[1]
            correct = (pred.view(-1) == target.view(-1)).long().sum().item()
            num_correct[i] += correct
            model_losses[i] += loss

    # end timing training
    time_end = time.time()

    N = len(test_loader.dataset)
    loss_str = ', '.join(['dev-{}: {:.4f}'.format(i, loss / N)
                        for i, loss in enumerate(model_losses[:-1])])
    acc_str = ', '.join(['dev-{}: {:.4f}%'.format(i, 100. * (correct / N))
                        for i, correct in enumerate(num_correct[:-1])])
    print('Inference elapsed in {:0f}s'.format(time_end-time_start))
    print('Test  Loss:: {}, cloud-{:.4f}'.format(loss_str, model_losses[-1] / N))
    print('Test  Acc.:: {}, cloud-{:.4f}%'.format(acc_str, 100. * (num_correct[-1] / N)))

    return model_losses, num_correct


def train_model(model, model_path, train_loader, test_loader, lr, epochs, num_devices):
    #ps = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(1, epochs):
        print('[Epoch {}/{}]'.format(epoch,epochs))
        train(model, train_loader, optimizer, num_devices)
        test(model, test_loader, num_devices)
        torch.save(model, model_path + 'ddnn.pth')
        torch.save(model.cloud_model, model_path + 'edge.pth')
        for i, device in enumerate(model.device_models):
            torch.save(device, model_path + 'dev' + str(i) + '.pth')
        scheduler.step()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Example')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--dataset', default='voc', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output', default='models/ddnn/',
                        help='output directory')
    parser.add_argument('--n_devices', default=5,
                        help='number of devices')

    args = parser.parse_args()

    # use cuda if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # seed???
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)


    train_dataset, train_loader, test_dataset, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, device)
    x, _ = train_loader.__iter__().next()
    in_channels = x.shape[2]
    out_channels = len(train_dataset.classes)

    # construct DDNN
    model = net(in_channels, out_channels, args.n_devices)

    # load on GPU
    model = model.to(device)

    # Data logging
    df = pd.DataFrame(columns=['Edge train loss'])

    # Run training
    train_model(model, args.output, train_loader, test_loader, args.lr, args.epochs, args.n_devices)