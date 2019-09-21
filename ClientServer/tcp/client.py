from socket import socket, AF_INET, SOCK_STREAM
import sys
import os
from tqdm import tqdm
from time import perf_counter, time
import pandas as pd
import argparse
import datetime
import struct
import math
import numpy as np

class TCPClient:

    def __init__(self):
        self.socket = socket(AF_INET, SOCK_STREAM) 
        self.HEAD_CODE = '<hhhd'
        self.HEAD_SIZE = struct.calcsize(self.HEAD_CODE)

    def start(self, host, port, chunk_size, image_paths, log, timeout=0):
        img_delay = []
        img_sizes = []

        with self.socket as s:
            if timeout:
                s.settimeout(timeout) # set the overall transmission timeout as per Python 3.5^
            
            print('Socket: Connecting to host: {}, port {}'.format(host, port))
            s.connect((host, port)) 



            for img_id, img_path in enumerate(tqdm(image_paths, desc='Sending {} images'.format(len(image_paths)))):
                

                # calculate required number of chunks
                img_info = os.stat(img_path)
                img_size = img_info.st_size
                payload_size = chunk_size# - self.HEAD_SIZE
                num_chunk = math.ceil(img_size / payload_size)

                s.sendall(img_size.to_bytes(4, byteorder='big'))

                delay = 0
                with open(img_path, 'rb') as img:
                    for chunk_id in range(num_chunk):
                        payload = img.read(payload_size)

                        # struct data: preamble, img_id, chunk_id, timestamp, payload
                        #head = struct.pack(self.HEAD_CODE, img_id, num_chunk, chunk_id, time())

                        # send data
                        tic = perf_counter()
                        #s.send(head + payload)
                        s.send(payload)
                        delay += perf_counter() - tic

                img_delay.append(delay)
                img_sizes.append(img_size)
                
                s.recv(3)
                
        print('Socket: Close')

        return img_delay, img_sizes
    
    def stop(self):
        self.socket.close()
        print('Socket closed')

    def log(self,name, dataframe):
        dataframe.to_csv('log/client/' + name + '.csv')

def main(args):
    client = TCPClient()

    try:
        img_paths = []
        if os.path.exists('ilsvrc2012_test.txt'):
            with open('ilsvrc2012_test.txt') as f:
                for x in f:
                    img_path = os.path.join(args.image_path, x.split()[0])
                    img_paths.append(img_path)
        else:
            images = os.listdir(args.image_path)
            for i in images:
                img_paths.append(os.path.join(args.image_path, i))


        delays, sizes = client.start(args.host, args.port, args.buffer_size, img_paths, args.log)

        n_images = len(img_paths)
        time_elapsed = sum(delays)
        tot_size = sum(sizes) / 1e6
        bit_size = (tot_size+4+3)*8

        stats = [{
            'n_transmissions': n_images, 
            'total_time (s)': time_elapsed, 
            'avg_time (s)': time_elapsed / n_images, 
            'total size (Mb)': tot_size,
            'avg. size (Mb)': tot_size / n_images,
            'data rate (Mbit/s)' : bit_size / time_elapsed
        }]
        df = pd.DataFrame(stats)

        if args.print:
            print(df)
        
        if args.log is not False:
            client.log(args.name + '_'+ str(args.buffer_size) + '_' + str(args.number), df)
            np.save('log/client/' + args.name + '_'+ str(args.buffer_size) + '_' + str(args.number), {'img_delay': delays, 'img_size': sizes})
            print('Test results have been saved.')

    except KeyboardInterrupt:
        client.stop()



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='TCP server')
    parser.add_argument('--name', default='client_run_name', help='Give a name to run')
    parser.add_argument('--number', default='1', help='run number')
    parser.add_argument('--host', default='127.0.0.1', help='host ip')
    parser.add_argument('--port', type=int, default=65432, help='port')
    parser.add_argument('--buffer_size', type=int, default=4096,  help='buffer size')
    parser.add_argument('--image_path', default="'/home/nuc/Documents/Dataset/ILSVRC2012/ILSVRC2012_images_val'", help='path to images')
    parser.add_argument('--timeout', type=int, default=0, help='timeout setting for entire request')
    parser.add_argument('--log', default=True, help='log run results')
    parser.add_argument('--print', default=True, help='Console print output')
    args = parser.parse_args()
    main(args)

    



