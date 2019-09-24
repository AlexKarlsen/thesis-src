import os
import math
import socket
import argparse
import struct
import time
import numpy as np

EDGE_PORT = 5001
CLIENT_PORT = 5002
PREAMBLE = 'ST'
HEAD_CODE = '<hhhd'
HEAD_SIZE = struct.calcsize(HEAD_CODE)

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge_addr', default='localhost', type=str, help='destination ipv4 address')
    parser.add_argument('--chunk_size', default=4096, type=int, help='in bytes')
    parser.add_argument('--save', default=False)
    args = parser.parse_args()

    return args

def udp_send(edge_addr, img_paths, chunk_size):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(5)
        idx = 0
        img_delay = []

        for img_id, img_path in enumerate(img_paths):
            while True:
                sock.sendto(img_id.to_bytes(4, 'big'), edge_addr)
                try:
                    msg, _ = sock.recvfrom(4)
                    if msg.decode('utf8') == 'REDY':
                        break
                except socket.timeout:
                    continue

            # calculate required number of chunks
            img_info = os.stat(img_path)
            img_size = img_info.st_size
            payload_size = chunk_size - HEAD_SIZE
            num_chunk = math.ceil(img_size / payload_size)

            delay = 0
            with open(img_path, 'rb') as img:
                for chunk_id in range(num_chunk):
                    payload = img.read(payload_size)

                    # struct data: preamble, img_id, chunk_id, timestamp, payload
                    head = struct.pack(HEAD_CODE, img_id, num_chunk, chunk_id, time.time())

                    # send data
                    tic = time.perf_counter()
                    sock.sendto(head + payload, edge_addr)
                    delay += time.perf_counter() - tic
                    idx += 1
                    #print('The {}-th chunk is successfully sent.'.format(chunk_id + 1))

                print('The {}-th image is successfully sent. Data rate is {} MB/s'.format(img_id+1, img_size/delay/1e6))
            img_delay.append(delay)
        print('Send {} packets in total.'.format(idx))

    return img_delay

if __name__ == '__main__':
    args = argument()

    img_paths = []
    with open('ilsvrc2012_test.txt') as f:
        for x in f:
            img_path = os.path.join('/home/nuc/Documents/Dataset/ILSVRC2012/ILSVRC2012_images_val', x.split()[0])
            img_paths.append(img_path)

    edge_addr = (args.edge_addr, EDGE_PORT)
    #get_clock_drift(edge_addr)

    img_delay = udp_send(edge_addr, img_paths, args.chunk_size)
    print('UDP test completed.')

    if args.save is not False:
        results_path = 'client_' + args.save
        np.save(results_path, {'img_delay': img_delay})
        print('Test results have been saved.')
