from socket import socket, AF_INET, SOCK_STREAM
import torch
from torchvision import transforms
import argparse
from dnn.dnn_runner import predictor
from PIL import Image
import io
import json
import torch.nn.functional as F

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

        connection.sendall(json.dumps(msg).encode('utf-8'))

    def receive(self, conn, buffer_size):
        # receive image size info
        size = conn.recv(4)

        if not size:
            return False

        data = b'' # dataholder
        size = int.from_bytes(size, byteorder='big') # convert byte to int
        print(size)
        while size > 0:
            part = conn.recv(buffer_size) # receive chunk
            data += part # append chunk
            size -= len(part) # substract from size to track progres
            print(size)
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


    model.eval()
    with torch.no_grad():
        
        s.start(args.host, args.port, args.buffer_size)
        connection = s.await_connection()
        while True:
            data = s.receive(connection, args.buffer_size)
            if data == False:
                print('Done receiving')
                return
            image = Image.open(io.BytesIO(data))
            image = transforms.ToTensor()(image)
            image = normalize(image)
            #img = Variable(img, requires_grad=False)
            image = image.unsqueeze(0)
            if torch.cuda.is_available():
                image = image.cuda()
            myPredictor = predictor(model, image, args.model_type)
            for ex in range(len(myPredictor.exits)):
                pred = next(myPredictor)
                score = F.softmax(pred, dim=1).max(1)[0].item()
                pred = pred.data.max(1, keepdim=True)[1].item()
                msg = {
                    'exit': ex,
                    'prediction': pred,
                    'confidence': score
                }
                s.send(connection, msg)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Intelligence Server')
    parser.add_argument('--model_path', default='models/b-resnet/miniimagenet_100_20191023-162944_model.pth',
                        help='output directory')

    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=23456)
    parser.add_argument('--buffer-size', default=4096)
    parser.add_argument('--model-type', default='b-resnet')
    args = parser.parse_args()
    main(args)