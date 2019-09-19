from socket import socket, AF_INET, SOCK_STREAM
import sys
from tqdm import tqdm
from time import perf_counter
import pandas as pd
import argparse

class TCPServer:
    def __init__(self):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.df = pd.DataFrame(columns=[
            'n_transmissions', 
            'total_time (s)', 
            'avg_time (s)', 
            'total size (Mb)', 
            'avg. size (Mb)', 
            'data rate (Mbit/s)'
            ])

    def saveImage(self, filename, data):
        with open('received/' + filename + '.jpg', 'wb') as i:
            i.write(data)

    def start(self, host, port, buffer_size, save=False):
        with self.socket as s:
            s.bind((host, port))
            print('Socket: Bind to ({},{})'.format(host, port))
            s.listen()
            print('Socket: Listen')
            
            data_arr = []
                
            conn, addr = s.accept()
            while True:
                
                with conn:
                    print('Socket: Connected by', addr)

                    tot_size = 0
                    n_images = 0
                    time_elapsed = 0
                    delays = []

                    while True:
                        # receive image size info
                        size = conn.recv(4)
                        if not size:
                            break
                        size = int.from_bytes(size, byteorder='big') # convert byte to int
                        tot_size += size # add size to total counter
                        
                        data = b'' # dataholder

                        with tqdm(total=size, unit='B', unit_scale=True, leave=False) as pbar:
                            delay = 0
                            while size > 0:
                                pbar.set_description(desc='Image #' + str(n_images))
                                tic = perf_counter() # timing
                                part = conn.recv(buffer_size) # receive chunk
                                data += part # append chunk
                                size -= len(part) # substract from size to track progres
                                delay += perf_counter() - tic # timing end

                                pbar.update(len(part))

                            data_arr.append(data)
                            delays.append(delay)
                            
                            
                            conn.sendall(b'ack')
                            n_images += 1

                tot_size = tot_size / 1e6
                time_elapsed = sum(delays)

                stats = {
                    'n_transmissions': n_images, 
                    'total_time (s)': time_elapsed,
                    'avg_time (s)' : time_elapsed/n_images,
                    'total size (Mb)': tot_size,
                    'avg. size (Mb)': tot_size / n_images,
                    'data rate (Mbit/s)' : ((tot_size+3+4)*8)/time_elapsed
                }

                self.df = self.df.append(stats, ignore_index=True)
                print(self.df)
                return data_arr

    def stop(self):
        self.socket.close()
        self.df.to_csv('log/1.csv')
        print('Socket closed')


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='TCP server')
    parser.add_argument('--host', default='127.0.0.1', help='host ip')
    parser.add_argument('--port', type=int, default=65432, help='port')
    parser.add_argument('--buffer_size', type=int, default=4096,  help='buffer size')
    parser.add_argument('--save', action='store_true', help='saving images on server')
    args = parser.parse_args()

    server = TCPServer()
    try:
        data = server.start(args.host, args.port, args.buffer_size, save=args.save)
        if args.save:
            for i, d in enumerate(data):
                server.saveImage(str(i), d)
    except KeyboardInterrupt:
        server.stop()

    
    