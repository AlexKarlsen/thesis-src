from socket import socket, AF_INET, SOCK_STREAM
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

class client():
    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket(AF_INET, SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host, port, timeout=0):
        self.sock.connect((host, port))
        print('Socket: Connecting to host: {}, port {}'.format(host, port))
        if timeout is not 0:
            self.sock.settimeout(timeout) # set the overall transmission timeout as per Python 3.5^

    def send(self, data):
        data = data[0,:,:,:]
        data = transforms.ToPILImage('RGB')(data)
        buf = io.BytesIO()
        data.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        self.sock.sendall(len(byte_im).to_bytes(4, byteorder='big'))
        self.sock.sendall(byte_im)

    def receive(self):
        # receive image size info
        data = self.sock.recv(1024)
        return json.loads(data.decode('utf-8'))


def main(args):
    # use cuda if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.n_classes, device)

    if args.edge_setting == 'collaborative':
        if device.type == 'cuda':
            torch.cuda.manual_seed(args.seed)
            model = torch.load(args.model_path)
        else:
            model = torch.load(args.model_path, map_location=torch.device('cpu'))

    c = client()
    c.connect(args.host, args.port)

    results = []
    for sample, (data, target) in enumerate(tqdm(test_loader, leave=False, unit='batch')):

        time_start = perf_counter()
        c.send(data)
        
        nExits = 4
        if args.model_type == 'msdnet':
            nExits = 5

        for _ in range(nExits):
            pred = c.receive()
            pred['target'] = target.view(-1).item()
            pred['time'] = (perf_counter() - time_start) * 1000
            pred['correct'] = (pred['prediction']==pred['target'])
            pred['sample'] = sample
            results.append(pred)
    log = pd.DataFrame(results)
    log.to_csv(args.name + '.csv')


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Intelligence Client')
    parser.add_argument('--name', default='edge intelligence', help='run name')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--dataset', default='miniimagenet-test-only-no-normalize', help='dataset name')
    parser.add_argument('--n-classes', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_path', default='models/b-densenet/miniimagenet_100_20191018-165914_model.pth',
                        help='output directory')
    parser.add_argument('--edge-setting', default='edge-only')
    parser.add_argument('--local-exits', default=0)

    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=23456)
    parser.add_argument('--model-type', default='resnet')
    args = parser.parse_args()
    main(args)