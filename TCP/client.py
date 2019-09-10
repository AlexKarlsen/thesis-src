from socket import socket, AF_INET, SOCK_STREAM
import sys
import os
from tqdm import tqdm
from time import perf_counter
import pandas as pd
import argparse

class TCPClient:
    def __init__(self):
        self.socket = socket(AF_INET, SOCK_STREAM) 

    def loadImage(self, filename):
        return open(filename, 'rb').read()

    def start(self, host, port, buffer_size, image_path, timeout=0, save=False):
        tot_size = 0
        files = os.listdir(image_path)
        with self.socket as s:
            if timeout:
                s.settimeout(timeout) # set the overall transmission timeout as per Python 3.5^
            time_start = perf_counter()
            print('Socket: Connecting to host: {}, port {}'.format(host, port))
            s.connect((host, port)) 

            for obj in tqdm(files, desc='Sending {} images'.format(len(files))):
                filename = os.path.join(image_path, obj)
                size = os.path.getsize(filename)
                tot_size += size
                s.sendall(size.to_bytes(4, byteorder='big'))
                
                tmp = self.loadImage(filename)
                s.sendall(tmp)
                # Should we confirm tx?
                # data = 
                s.recv(buffer_size) # await response from server
                #print('Received: {}'.format(data.decode('utf8')))

            time_end = perf_counter()
        
        print('Socket: Close')
        time_elapsed = time_end - time_start
        tot_size = tot_size / 1000000

        return {
            'n_transmissions': len(files), 
            'total_time (s)': time_elapsed, 
            'avg_time (s)': time_elapsed / len(files), 
            'total size (Mb)': tot_size,
            'avg. size (Mb)': tot_size / len(files),
            'data rate (Mbit/s)' : (tot_size*8)/time_elapsed
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
    parser.add_argument('--image_path', default="images", help='path to images')
    parser.add_argument('--timeout', type=int, default=0, help='timeout setting for entire request')
    parser.add_argument('--save', action='store_true', help='saving images on server')
    args = parser.parse_args()

    client = TCPClient()

    try:
        stats = client.start(args.host, args.port, args.buffer_size, args.image_path, timeout=args.timeout, save=args.save)
        df = pd.DataFrame(columns=['n_transmissions', 'total_time (s)', 'avg_time (s)', 'total size (Mb)', 'avg. size (Mb)', 'data rate (Mbit/s)'])
        df = df.append(stats, ignore_index=True)
        print(df)
    except KeyboardInterrupt:
        client.stop()

    



