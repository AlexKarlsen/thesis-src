import argparse
import os
import socket
import time
import struct
import numpy as np
import packet_process as pp

class Configuration():
    EDGE_PORT = 5001
    CLIENT_PORT = 5002
    TIMEOUT = 1

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

def udp_recv(cfg, edge_addr, chunk_size):
    packet_list = []
    time_list = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(cfg.TIMEOUT)
        sock.bind(edge_addr)

        while True:
            try:
                msg, client_addr = sock.recvfrom(4)
            except socket.timeout:
                continue

            img_id = int.from_bytes(msg, 'big')
            msg = 'REDY'.encode('utf8')
            sock.sendto(msg, client_addr)
            #print('Get ready to receive data.')
            while True:
                try:
                    data, _ = sock.recvfrom(chunk_size)
                    tic = time.time()

                    try:
                        head = struct.unpack(cfg.HEAD_CODE, data[:cfg.HEAD_SIZE])
                        time_list.append(tic)
                        packet_list.append(data)
                        if head[2] + 1 == head[1]:
                            break
                    except struct.error:
                        pass

                except KeyboardInterrupt:
                    break
                except socket.timeout:
                    break
            if img_id == 1000:
                break

    print('Receiving completed.')
    return packet_list, time_list

if __name__ == '__main__':
    cfg = Configuration()
    args = argument()
    edge_addr = (args.edge_addr, cfg.EDGE_PORT)

    # udp receive
    packet_list, time_list = udp_recv(cfg, edge_addr, args.chunk_size)
    print('Receive {} packets in total.'.format(len(packet_list)))

    # unpack received data
    payload_list = []
    for packet, recv_time in zip(packet_list, time_list):
        out = pp.get_payload_from_packet(cfg, packet, recv_time)
        payload_list.append(out)

    # assemble packet to image bytes list
    img_byte_list, delay_list = pp.img_byte_assemble(cfg, payload_list, args.chunk_size)

    # calculate metrics and save results
    results = pp.calculate_udp_perform(img_byte_list, delay_list)

    # save results
    if args.save is not False:
        results_path = os.path.join('results', 'edge_' + args.save)
        np.save(results_path, results)
        print('Test results have been saved.')
