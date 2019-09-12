from socket import socket, AF_INET, SOCK_STREAM
import sys
import os
from tqdm import tqdm
from time import perf_counter
import pandas as pd
import argparse
import datetime

class TCPClient:
    def __init__(self):
        self.socket = socket(AF_INET, SOCK_STREAM) 

    def loadImage(self, filename):
        return open(filename, 'rb').read()

    def start(self, host, port, buffer_size, image_path, timeout=0, save=False):
        tot_size = 0
        time_elapsed = 0
        files = image_path
        with self.socket as s:
            if timeout:
                s.settimeout(timeout) # set the overall transmission timeout as per Python 3.5^
            
            print('Socket: Connecting to host: {}, port {}'.format(host, port))
            s.connect((host, port)) 

            for obj in tqdm(files, desc='Sending {} images'.format(len(files))):
                time_start = perf_counter()
                filename = obj
                size = os.path.getsize(filename)
                tot_size += size
                s.sendall(size.to_bytes(4, byteorder='big'))
                
                tmp = self.loadImage(filename)
                s.sendall(tmp)
                # Should we confirm tx?
                # data = 
                
                s.recv(3) # await response from server
                #print('Received: {}'.format(data.decode('utf8')))
                time_end = perf_counter()
                time_elapsed += (time_end - time_start)
                
        print('Socket: Close')
        
        tot_size = tot_size / 1000000

        return {
            'n_transmissions': len(files), 
            'total_time (s)': time_elapsed, 
            'avg_time (s)': time_elapsed / len(files), 
            'total size (Mb)': tot_size,
            'avg. size (Mb)': tot_size / len(files),
            'data rate (Mbit/s)' : ((tot_size+4+3)*8)/time_elapsed
            }
    
    def stop(self):
        self.socket.close()
        print('Socket closed')


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='TCP server')
    parser.add_argument('--host', default='127.0.0.1', help='host ip')
    parser.add_argument('--port', type=int, default=65432, help='port')
    parser.add_argument('--buffer_size', type=int, default=4096,  help='buffer size')
    parser.add_argument('--image_path', default="'/home/nuc/Documents/Dataset/ILSVRC2012/ILSVRC2012_images_val'", help='path to images')
    parser.add_argument('--timeout', type=int, default=0, help='timeout setting for entire request')
    parser.add_argument('--save', action='store_true', help='saving images on server')
    args = parser.parse_args()

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
        stats = client.start(args.host, args.port, args.buffer_size, img_paths, timeout=args.timeout, save=args.save)
        df = pd.DataFrame(columns=['n_transmissions', 'total_time (s)', 'avg_time (s)', 'total size (Mb)', 'avg. size (Mb)', 'data rate (Mbit/s)'])
        df = df.append(stats, ignore_index=True)
        print(df)
        now = datetime.datetime.now()
        df.to_csv(str(datetime.datetime.timestamp(now))+'.csv')
    except KeyboardInterrupt:
        client.stop()

    



