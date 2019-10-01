import os
import numpy as np
import pyshark
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
#from dnn import classification
from plot_utility import *

"""
This is the script to analyze the results of the UDP and TCP test between NUC and JETSON TX2

For UDP results (udp_20190916):

    The delay to transmit each image is obtained by 'client_%scenario_%chunksize_%no.npy'
    
    The number of packet loss in each image is obtained by 'edge_%scenario_%chunksize_%no.npy'
    
    The inference accuracy of the received images is obtained by by 'edge_%scenario_%chunksize_%no.npy'
    
    The pcap-ng file captured by wireshark at the client side has no specific usage for analysis
    
For TCP results (tcp_20190912):
    
    '%scenario_%no_%chunksize.npy' is used for obtianing the delay histogram of each image transmission
    
    '%scenario_%no_%chunksize.pcapng' is used for statistic of the number of retransmission and delay check
    
    '%scenario_%no_%chunksize.csv' saved the averaged metrics of each tcp trail
    
"""
class Configuration():
    scenario = ['meetingroom', 'kitchen', 'corridor']
    sc_color = ['r', 'g', 'b']

    chunk_size = [4096]
    size_ls = ['-', '--']

    num_trail = 5
    protocol = 'tcp'

def get_tcp_delay_retransmission_info(results_path, size, sc, idx):
    """
    Args:
        size: chunk_size, e.g. 1024, 4096
        sc: scenario, e.g. 'meetingroom', 'kitchen', 'corridor'
        idx: index of trails for each setting

    tcp retransmission info      --> pkt.tcp.analysis_retransmission: 'This frame is a (suspected) retransmission'
    pkt length                   --> pkt.length / pkt.frame_info.cap_len
    payload length               --> pkt.tcp.len

    timestamp                    --> pkt.sniff_timestamp / pkt.frame_info.time_epoch
    delay from last captured pkt --> pkt.frame_info.time_delta

    each image transmission starts from a info message with 4-bytes payload, and ends by a pkt with payload of below 1448 bytes
    """

    path = os.path.join(results_path, sc, '{}_{}_{}.pcapng'.format(sc, idx, size))

    delay_list = []
    retran_list = []
    end_list = []
    overhead_list = []

    if not os.path.exists(path):
        print('There is no such file', path)
    else:
        print(path)
        pkt_data = pyshark.FileCapture(path)

        retran = 0 # the number of retransmissions per image
        overhead = 0 # number of frame used for tcp control signals
        start = 0. # denote the starting timestamp per image, i.e., img info frame
        end = 0. # denote the ending timestamp per image, i.e., the last img frame before 3-byte response
        end_id = 0 #
        img_flag = False # denote if the period of image transmission
        img_idx = 0
        for idx, pkt in enumerate(pkt_data):
            pkt_len = int(pkt.tcp.len)

            if pkt_len == 4 and img_flag == False:
                start = float(pkt.frame_info.time_epoch)
            elif pkt_len == 3 and img_flag == True:
                img_flag = False
            elif pkt_len == 0 and img_flag == True:
                overhead += 1
            elif pkt_len > 4:
                end = float(pkt.frame_info.time_epoch)
                end_id = idx + 1

                if img_flag == False:
                    img_flag = True
                if 'analysis_retransmission' in pkt.tcp.field_names:
                    retran += 1
                    overhead += 1
            else:
                pass

            if img_flag == False and end != 0.:
                delay_list.append((end - start))
                retran_list.append(retran)
                overhead_list.append(overhead)
                end_list.append(end_id)

                end = 0.
                retran = 0
                overhead = 0

                img_idx += 1
        print('{} images are analyzed.'.format(img_idx))

    if len(delay_list) > 1001:
        gap = len(delay_list) - 1001
        for g in range(gap):
            delay_list.remove(delay_list[-1])
            retran_list.remove(retran_list[-1])
            overhead_list.remove(overhead_list[-1])

    return delay_list, retran_list, overhead_list

def get_udp_lossrate_per_img(results_path, size, sc, idx):
    path = os.path.join(results_path, sc, 'edge_{}_{}_{}.npy'.format(sc, size, idx))
    data = np.load(path, allow_pickle=True).item()

    return  data['loss_rate'], data['recv_img_byte']

def get_udp_delay_per_img(results_path, size, sc, idx):
    if sc == 'meetingroom' and size == 1024:
        path = os.path.join(results_path, sc, 'covered', 'client_{}_{}_{}.npy'.format(sc, size, idx))
    else:
        path = os.path.join(results_path, sc, 'client_{}_{}_{}.npy'.format(sc, size, idx))
    data = np.load(path, allow_pickle=True).item()['img_delay'] # in seconds

    return data

def udp_analysis(cfg, results_path):

    delay_list = []
    loss_rate = []
    img_byte = []
    accuracy = []

    for size in cfg.chunk_size:
        delay_list.append([])
        loss_rate.append([])
        img_byte.append([])
        accuracy.append([])

        for sc in cfg.scenario:
            delay_list[-1].append([])
            loss_rate[-1].append([])
            img_byte[-1].append([])

            for idx in range(cfg.num_trail):
                delay_list[-1][-1] += get_udp_delay_per_img(results_path, size, sc, idx+1)
                lr, ib = get_udp_lossrate_per_img(results_path, size, sc, idx+1)

                loss_rate[-1][-1] += lr
                img_byte[-1][-1] += ib

            #accuracy[-1].append(classification.eval(img_byte[-1][-1], num_trail))

            #print(np.nonzero(loss_rate[-1][-1]))

    # Delay per image
    draw_histogram_delay_per_img(cfg, np.array(delay_list)*1000)
    draw_average_delay_per_img(cfg, np.array(delay_list)*1000)

    # Number of packet loss per image
    draw_average_lossrate_per_img(cfg, np.array(loss_rate)*100)
    draw_max_lossrate_per_img(cfg, np.array(loss_rate))

    # Average inference accuracy
    #draw_inference_accuracy(cfg, accuracy[:][:][0], 1)
    #draw_inference_accuracy(cfg, accuracy[:][:][1], 5)

    # check corrupted images
    # with open('corrupt_6.jpeg', 'wb') as f:
    #     f.write(img_byte[1][2][4802])
    # print(loss_rate[0][1][662], loss_rate[0][1][752],
    #       loss_rate[0][2][927], loss_rate[0][2][1001], loss_rate[0][2][1240],
    #       loss_rate[1][2][4802])

def tcp_analysis(cfg, resutls_path):
    delay_list = []
    retran_list = []
    overhead_list = []

    for size in cfg.chunk_size:
        delay_list.append([])
        retran_list.append([])
        overhead_list.append([])

        for sc in cfg.scenario:
            delay_list[-1].append([])
            retran_list[-1].append([])
            overhead_list[-1].append([])

            measure_trails = 0
            for idx in range(cfg.num_trail):
                if measure_trails == 4:
                    break
                else:
                    dl, rl, ol = get_tcp_delay_retransmission_info(results_path, size, sc, idx + 1)
                    if len(dl) != 0:
                        measure_trails += 1
                        delay_list[-1][-1] += dl
                        retran_list[-1][-1] += rl
                        overhead_list[-1][-1] += ol
    #
    # np.save('tcp_wireshark_analysis_delay', delay_list)
    # np.save('tcp_wireshark_analysis_retrans', retran_list)
    # np.save('tcp_wireshark_analysis_overhead', overhead_list)

    # delay_list = np.load('tcp_wireshark_analysis_delay.npy', allow_pickle=True)
    # retran_list = np.load('tcp_wireshark_analysis_retrans.npy', allow_pickle=True)
    # overhead_list = np.load('tcp_wireshark_analysis_overhead.npy', allow_pickle=True)

    print(np.sum(retran_list[0][2][:1001]))
    # draw_average_delay_per_img(cfg, np.array(delay_list)*1000)
    # draw_histogram_delay_per_img(cfg, np.array(delay_list)*1000)

    # draw_average_retransmission_per_img(cfg, np.array(retran_list))
    # draw_max_retransmission_per_img(cfg, np.array(retran_list))

    draw_average_overhead_per_img(cfg, np.array(overhead_list))
    draw_histogram_overhead_per_img(cfg, np.array(overhead_list))

if __name__ == '__main__':
    cfg = Configuration()

    # The root directory of results
    if cfg.protocol == 'udp':
        results_path = os.path.join('test_results', cfg.protocol+'_20190916')
        udp_analysis(cfg, results_path)
    elif cfg.protocol == 'tcp':
        results_path = os.path.join('test_results', cfg.protocol+'_20190912')
        tcp_analysis(cfg, results_path)
    else:
        raise 'Protocol input error.'