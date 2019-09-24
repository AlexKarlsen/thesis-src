from socket import socket, AF_INET, SOCK_STREAM
import sys
from tqdm import tqdm
from time import perf_counter
import pandas as pd
import argparse
import numpy as np

class TCPServer:
    def __init__(self):
        self.socket = socket(AF_INET, SOCK_STREAM)
        

    def saveImage(self, filename, data):
        with open('received/' + filename + '.jpg', 'wb') as i:
            i.write(data)

    def start(self, host, port, buffer_size, save=False):
        with self.socket as s:
            s.bind((host, port))
            print('Socket: Bind to ({},{})'.format(host, port))
            s.listen()
            print('Socket: Listen')
            
            images = []
            delays = []
            sizes = []

            conn, addr = s.accept()
            while True:
                
                with conn:
                    print('Socket: Connected by', addr)

                    n_images = 0
                    while True:
                        # receive image size info
                        size = conn.recv(4)
                        if not size:
                            break
                        size = int.from_bytes(size, byteorder='big') # convert byte to int
                        sizes.append(size)
                        
                        data = b'' # dataholder

                        with tqdm(total=size, unit='B', unit_scale=True, leave=False) as pbar:
                            delay = 0
                            while size > 0:

                                #  try:
                                #     head = struct.unpack(cfg.HEAD_CODE, data[:cfg.HEAD_SIZE])
                                #     time_list.append(tic)
                                #     packet_list.append(data)
                                #     if head[2] + 1 == head[1]:
                                #         break
                                # except struct.error:
                                #     pass

                                pbar.set_description(desc='Image #' + str(n_images))
                                tic = perf_counter() # timing
                                part = conn.recv(buffer_size) # receive chunk
                                data += part # append chunk
                                delay += perf_counter() - tic # timing end
                                size -= len(part) # substract from size to track progres
                                

                                pbar.update(len(part))

                            images.append(data)
                            delays.append(delay)
                            
                            
                            conn.sendall(b'ack')
                            n_images += 1

                return images, delays, sizes

    def stop(self):
        self.socket.close()
        print('Socket closed')

    def log(self,name, dataframe):
        dataframe.to_csv('log/server/' + name + '.csv')

def main(args):
    server = TCPServer()
    try:
        images, delays, sizes = server.start(args.host, args.port, args.buffer_size, save=args.save)

        n_images = len(images)
        time_elapsed = sum(delays)
        tot_size = sum(sizes)
        tot_size = tot_size / 1e6

        stats = [{
                        'n_transmissions': n_images, 
                        'total_time (s)': time_elapsed,
                        'avg_time (s)' : time_elapsed/n_images,
                        'total size (Mb)': tot_size,
                        'avg. size (Mb)': tot_size / n_images,
                        'data rate (Mbit/s)' : ((tot_size+3+4)*8)/time_elapsed
                    }]

        df = pd.DataFrame(stats)

        if args.print:
            print(df)

        if args.log:
            server.log(args.name + '_'+ str(args.buffer_size) + '_' + args.number, df)
            np.save('log/server/' + args.name + '_'+ str(args.buffer_size) + '_' + str(args.number), {'img_delay': delays, 'img_size': sizes})
            print('Test results have been saved.')
        
        if args.save:
            for i, data in enumerate(images):
                server.saveImage('image_' + str(i), data)
    except KeyboardInterrupt:
        server.stop()



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='TCP server')
    parser.add_argument('--name', default='server_run_name', help='Give a name to run')
    parser.add_argument('--number', default='1', help='run number')
    parser.add_argument('--host', default='127.0.0.1', help='host ip')
    parser.add_argument('--port', type=int, default=65432, help='port')
    parser.add_argument('--buffer_size', type=int, default=4096,  help='buffer size')
    parser.add_argument('--save', action='store_true', help='saving images on server')
    parser.add_argument('--print', default=True, help='output print')
    parser.add_argument('--log', default=True, help="save log file")
    args = parser.parse_args()
    main(args)
    

    
    