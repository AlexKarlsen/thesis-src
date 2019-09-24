import os
import math
import struct
#import matplotlib.pyplot as plt
import numpy as np

def get_img_size(gt_path = '../client/ilsvrc2012_test.txt'):
    with open(gt_path) as gt:
        with open('ilsvrc2012_test_img_gt_byte.txt', 'w') as f:
            for x in gt:
                img_path = os.path.join('/home/nuc/Documents/Dataset/ILSVRC2012/ILSVRC2012_images_val', x.split()[0])
                img_info = os.stat(img_path)
                img_size = img_info.st_size

                sx = x.split()
                f.writelines(sx[0] + ' ' + sx[1] + ' ' + str(img_size) + '\n')

def get_payload_from_packet(cfg, packet, recv_time):
    head = struct.unpack(cfg.HEAD_CODE, packet[:cfg.HEAD_SIZE])
    payload = packet[cfg.HEAD_SIZE:]
    return [payload, head[0], head[2], head[3], recv_time]

def img_byte_assemble(cfg, payload_list, chunk_size):
    img_byte = []
    delay = []

    with open('ilsvrc2012_test_img_gt_byte.txt') as f:
        for x in f:
            payload_size = chunk_size - cfg.HEAD_SIZE
            num_chunk = math.ceil(int(x.split()[2]) / payload_size)
            img_byte.append(['EMPTY' for i in range(num_chunk)])
            delay.append([0. for i in range(num_chunk)])

    for pl in payload_list:
        img_byte[pl[1]][pl[2]] = pl[0]
        delay[pl[1]][pl[2]] = (pl[4] - pl[3]) * 1000 # in millisecond

    return img_byte, delay

def calculate_udp_perform(img_bytes, delays):
    drop_list = []
    drop_rate = []
    data_rate = []
    imgs = []

    for img_byte, delay in zip(img_bytes, delays):
        img = b''
        drop_num = 0
        img_delay = 0
        drop_idx = []

        for idx, (chunk, chunk_delay) in enumerate(zip(img_byte, delay)):
            if chunk == 'EMPTY':
                drop_num += 1
                drop_idx.append(idx+1)
            else:
                img_delay += chunk_delay
                img += chunk

        imgs.append(img)
        drop_rate.append(drop_num/len(img_byte))
        drop_list.append(drop_idx)

        if img_delay != 0:
            data_rate.append(len(img) / img_delay / 1000) # in MB/s

    avg_drop_rate = np.mean(drop_rate)
    avg_data_rate = np.mean(data_rate)

    print('The average packet loss rate is {}.'.format(avg_drop_rate))
    print('The average UDP throughput is {} MB/s'.format(avg_data_rate))

    return {'recv_img_byte': imgs, 'loss_idx': drop_list, 'loss_rate': drop_rate, 'throughput': data_rate}

# if __name__ == '__main__':
#     get_img_size()
# with open(save_path, 'wb') as img:
#     # receive preamble
#     preamble, _ = sock.recvfrom(start_size + head_size)
#     if preamble[:3].decode('utf8') == 'STA':
#         print('Receiving starts.')
#         # receive the entire size info
#         img_size = int.from_bytes(preamble[start_size:], 'big')
#         num_chunk = math.ceil(img_size / (chunk_size - head_size))
#         print('{} bytes data will be received with {} chunks.'.format(img_size, num_chunk))
#
#         # dropout chunk randomly
#         drop = x
#         # drop = random.randint(1,num_chunk+1)
#         print('The {}-th chunk will be dropped.'.format(drop))
#
#         data = []
#         counter = 0
#         while counter < num_chunk:
#             counter += 1
#             try:
#                 pkt, client_addr = sock.recvfrom(chunk_size)
#                 head = int.from_bytes(pkt[:head_size], 'big')
#                 # print('The {}-th chunk has been received with length of {} bytes.'.format(head+1, len(pkt)-head_size))
#                 if head == drop:
#                     #       payload = bytearray(chunk_size-head_size if counter < num_chunk-1
#                     #                           else img_size-(chunk_size-head_size)*(num_chunk-1))
#                     #       print(len(payload))
#                     # else:
#                     #       payload = pkt[head_size:]
#                     continue
#                 payload = pkt[head_size:]
#                 img.write(payload)
#                 data.append(payload)
#             except KeyboardInterrupt:
#                 break
#     print(len(b''.join(data)))
#     data_np = cv2.imdecode(np.frombuffer(b''.join(data), dtype=np.uint8), -1)
#     # cv2.imshow('Received image', data_np)
#     # cv2.waitKey()
#     cv2.imwrite('cv_recv.jpeg', data_np, [cv2.IMWRITE_JPEG_QUALITY, 95])
#     # orl = cv2.imread('../client/ILSVRC2012_val_00000028.JPEG')
#     # print(np.array_equal(data_np, orl))
#
#     print('Image receiving is completed!')