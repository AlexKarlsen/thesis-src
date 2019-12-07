from socket import socket, AF_INET, SOCK_STREAM
import argparse
import json
from time import perf_counter
import threading

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

    s = server()
    s.start(args.host, args.port, args.buffer_size)
    connection = s.await_connection()

    while True:
        data = s.receive(connection, args.buffer_size)
        # set the number of exits
        nExits = 4 if args.model_type is not 'msdnet' else 5
        # set range of exits
        if data == False:
            print('Done receiving')
            return
        
        msg = {
            'exit': 9,
            'prediction': [0, 65, 10, 88, 92],
            'confidence': [0.9997285008430481, 4.163753692409955e-05, 3.712219404405914e-05, 3.569274485926144e-05, 2.643913285282906e-05],
            'prediction time': 596.7068000000069,
            'preprocess time': 596.7068000000069,
            'rx-time': 596.7068000000069
        }

            #print('prediction time: {}'.format(time_end-time_start))

        # send intermediate results ###### maybe threading would help

        #threading._start_new_thread(s.send,(connection, msg))
        # unthreaded
        s.send(connection, msg)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Intelligence Server')
    parser.add_argument('--model_path', default='models/msdnet/msdnet_miniimagenet100.pth',
                        help='output directory')

    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=23456)
    parser.add_argument('--buffer-size', default=4096)
    parser.add_argument('--model-type', default='b-resnet')
    args = parser.parse_args()
    main(args)