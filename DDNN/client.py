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
import torch.nn.functional as F
from dnn.dnn_runner import predictor
import pickle
import cv2
import threading



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

    def sendIntermediateFeatures(self, data, compress=False):
        if compress:
            data = data.numpy()[0,:,:,:]
            data.tobytes()
            result, data = cv2.imencode('.jpg', data)
        else:
            #data = data.numpy()[0,:,:,:]
            #data = np.flip(data)
            #data = np.ndarray.tobytes(data)
            data = pickle.dumps(data)
        self.sock.sendall(len(data).to_bytes(4, byteorder='big'))
        self.sock.sendall(data)

    def receive(self):
        # receive image size info
        size = self.sock.recv(4)
        size = int.from_bytes(size, byteorder='big')
        data = self.sock.recv(size)
        return json.loads(data.decode('utf-8'))


def main(args):
    # use cuda if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.n_classes, device)

    c = client()
    c.connect(args.host, args.port)

    results = []
    nExits = 4 if args.model_type is not 'msdnet' else 5
    if args.edge_setting == 'collaborative':
        nExits = nExits - args.local_exits
        _, test_loader = datasets.get_dataset(args.dataset_root, 'miniimagenet-test-only', args.batch_size, args.n_classes, device)
    else:
        _, test_loader = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.n_classes, device)

    for sample, (data, target) in enumerate(tqdm(test_loader, leave=False, unit='batch')):

        result = {}
        if args.edge_setting == 'collaborative':
            
            if device.type == 'cuda':
                torch.cuda.manual_seed(args.seed)
                model = torch.load(args.model_path)
                data, target = data.cuda, target.cuda()
            else:
                model = torch.load(args.model_path, map_location=torch.device('cpu'))

            model.eval()
            with torch.no_grad():
                myPredictor = predictor(model, data, args.model_type)
                time_start = perf_counter()
                for ex in range(args.local_exits):
                    p = next(myPredictor)
                    score = F.softmax(p, dim=1).max(1)[0].item()
                    p = p.data.max(1, keepdim=True)[1].item()
                    pred = {
                        'exit': ex,
                        'prediction': p,
                        'confidence': score,
                        'time' : (perf_counter() - time_start) * 1000,
                        'correct' : p == target.view(-1).item(),
                        'sample': sample
                    }
                    results.append(pred)
                
                threading._start_new_thread(c.sendIntermediateFeatures,(myPredictor.data,))
                #c.sendIntermediateFeatures(myPredictor.data)
        else:
            time_start = perf_counter()
            c.send(data)
            time_sent = perf_counter()
            
        

        for _ in range(nExits):
            pred = c.receive()
            result['sample'] = sample
            result['exit'] = pred['exit']
            result['prediction'] = pred['prediction']
            result['scores'] = pred['confidence']
            result['target'] = target.view(-1).item()
            result['overall time'] = (perf_counter() - time_start) * 1000
            result['prediction time'] = pred['prediction time']
            result['tx time'] = (time_sent-time_start)*1000
            result['rx time'] = pred['rx-time']
            result['preprocess time'] = pred['preprocess time']
            result['correct'] = (pred['prediction'][0]==result['target'])
            try:
                result['index_top5'] = pred['prediction'].index(result['target'])
            except ValueError:
                result['index_top5'] = -1

            if args.log_to_console:
                print(result)
            
            results.append(result.copy())

        
        with open('edge_test/' +args.name +'.json', 'w') as f:
            json.dump(results, f)
    #log.to_csv(args.name + '.csv')


    
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
    parser.add_argument('--model-type', default='b-densenet')
    parser.add_argument('--edge-setting', default='edge-only')
    parser.add_argument('--local-exits', default=1)

    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=23456)

    parser.add_argument('--log-to-console', default=True)
    args = parser.parse_args()
    main(args)