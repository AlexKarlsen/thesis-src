import socket
import sys
from tqdm import tqdm
from time import perf_counter
import pandas as pd
import argparse

def saveImage(filename, data):
    with open('received/' + filename + '.jpg', 'wb') as i:
        i.write(data)

def runServer(host, port, buffer_size, save=False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print('Listening ({},{})'.format(host, port))

        conn, addr = s.accept()
        time_start = perf_counter()
        with conn:
            print('Connected by', addr)

            tot_size = 0
            n_images = 0

            while True:
                size = conn.recv(4)
                if not size:
                    break
                size = int.from_bytes(size, byteorder='big')
                tot_size += size
                data = b''

                with tqdm(total=size, unit='B', unit_scale=True, leave=True) as pbar:
                    while size > 0:
                        pbar.set_description(desc='Image #' + str(n_images))
                        part = conn.recv(buffer_size)
                        data += part
                        size -= len(part)

                        pbar.update(len(part))

                        if len(part) < buffer_size: # either 0 or end of data

                            #print('Received image #{}'.format(n_images))
                            if save:
                                saveImage(str(n_images), data)
                            n_images +=1

                            reply = 'server received image #{}'.format(n_images)
                            conn.sendall(bytes(reply.encode('utf8')))
                            break

    time_end = perf_counter()
    time_elapsed = time_end - time_start

    tot_size = tot_size / 1000000

    return {
        'n_transmissions': n_images, 
        'total_time (s)': time_elapsed,
        'avg_time (s)' : time_elapsed/n_images,
        'total size (Mb)': tot_size,
        'avg. size (Mb)': tot_size / n_images,
        'data rate (Mbit/s)' : (tot_size*8)/time_elapsed
        }


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='TCP server')
    parser.add_argument('--host', default='127.0.0.1', help='host ip')
    parser.add_argument('--port', type=int, default=65432, help='port')
    parser.add_argument('--buffer_size', type=int, default=4096,  help='buffer size')
    parser.add_argument('--save', action='store_true', help='saving images on server')
    args = parser.parse_args()

    stats = runServer(args.host, args.port, args.buffer_size, args.save)

    df = pd.DataFrame(columns=['n_transmissions', 'total_time (s)', 'avg_time (s)', 'total size (Mb)', 'avg. size (Mb)', 'data rate (Mbit/s)'])
    df = df.append(stats, ignore_index=True)
    print(df)

    
    