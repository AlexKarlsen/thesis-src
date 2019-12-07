from socket import socket, AF_INET, SOCK_STREAM
import torch
from torchvision import transforms
import argparse
from dnn.dnn_runner import predictor
from PIL import Image
import io
import json
import torch.nn.functional as F
import pickle
import numpy as np
import cv2
import threading
from time import perf_counter

class server:
    def __init__(self):
       self.sock = socket(AF_INET, SOCK_STREAM) 

    def start(self, host, port, buffer_size):
        self.sock.bind((host, port))
        print('Socket: Bind to ({},{})'.format(host, port))
        self.sock.listen()
        print('Socket: Listen')
            
    def await_connection(self):
        while True:
            connection, address = self.sock.accept()
            print('Socket: Connected by', address)
            return connection
               
    
    def send(self, connection, msg):
        msg = json.dumps(msg).encode('utf-8')
        msg_len = len(msg)
        connection.sendall(msg_len.to_bytes(4, byteorder='big'))
        connection.sendall(msg)

    def receive(self, conn, buffer_size):
        # receive image size info
        size = conn.recv(4)

        if not size:
            return False

        data = b'' # dataholder
        size = int.from_bytes(size, byteorder='big') # convert byte to int
        while size > 0:
            part = conn.recv(buffer_size) # receive chunk
            data += part # append chunk
            size -= len(part) # substract from size to track progres
        return data

def main(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    s = server()

    # use cuda if available else use cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        model = torch.load(args.model_path)
    else:
        model = torch.load(args.model_path, map_location=torch.device('cpu'))

    # set the number of exits
    nExits = 4 if args.model_type is not 'msdnet' else 5
    myPredictor = predictor(model, None, args.model_type)
    # put model into eval mode and set require no grad
    model.eval()
    with torch.no_grad():
        # establish connection
        s.start(args.host, args.port, args.buffer_size)
        connection = s.await_connection()

        # while data is being received
        while True:
            data = s.receive(connection, args.buffer_size)
            
            # stop when flag is sent
            if data == False:
                print('Done receiving')
                return
            # set the edge mode
            if args.edge_setting == 'collaborative':
                #data = np.frombuffer(data, dtype='float32')
                #data = np.resize(data,(56,56,256))

                # load intermediate feature
                data = pickle.loads(data)

                # set range of exits
                exits = np.arange(args.local_exits, nExits, 1, dtype=np.uint8)
            else:
                # load image from bytes and perform preprocessing
                data = Image.open(io.BytesIO(data))
                data = transforms.ToTensor()(data)
                data = normalize(data)
                data = data.unsqueeze(0)

                # set range of exits
                exits = range(nExits) 

            # load data to gpu if available
            if device.type == 'cuda':
                data = data.cuda()
    
            myPredictor.data = data


            #print('preprocess time {}'.format(preprocess_time))

            # run the test
            time_start = perf_counter()
            for ex in exits:
                # predict
                myPredictor.counter = ex
                
                pred = next(myPredictor)
                pred = pred.cpu()
                score = F.softmax(pred, dim=1)
                prob, pred = torch.topk(score, k=5)
                prob, pred = prob.numpy()[0].tolist(), pred.numpy()[0].tolist() 
                prediction_time = (perf_counter() - time_start)*1000 
                # create msg
                msg = {
                    'exit': int(ex),
                    'prediction': pred,
                    'confidence': prob,
                    'prediction time': prediction_time,
                    #'preprocess time': preprocess_time,
                    #'rx-time': rx_time
                }

                #print('prediction time: {}'.format(time_end-time_start))

                # send intermediate results ###### maybe threading would help
                threading._start_new_thread(s.send,(connection, msg,))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Intelligence Server')
    parser.add_argument('--model_path', default='models/msdnet/msdnet_miniimagenet100.pth',
                        help='output directory')

    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=23456)
    parser.add_argument('--buffer-size', default=4096)
    parser.add_argument('--model-type', default='msdnet')
    parser.add_argument('--edge-setting', default='edge-only')
    parser.add_argument('--local-exits', default=0)
    args = parser.parse_args()
    main(args)