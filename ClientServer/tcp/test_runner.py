import client
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TCP server')
    parser.add_argument('--name', default='client_run_name', help='Give a name to run')
    parser.add_argument('--number', type=int, default=1, help='run number')
    parser.add_argument('--host', default='127.0.0.1', help='host ip')
    parser.add_argument('--port', type=int, default=65432, help='port')
    parser.add_argument('--buffer_size', type=int, default=4096,  help='buffer size')
    parser.add_argument('--image_path', default="'/home/nuc/Documents/Dataset/ILSVRC2012/ILSVRC2012_images_val'", help='path to images')
    parser.add_argument('--timeout', type=int, default=0, help='timeout setting for entire request')
    parser.add_argument('--log', default=True, help='log run results')
    parser.add_argument('--print', default=True, help='Console print output')
    args = parser.parse_args()
    for i in range(5):
        client.main(args)
        args.number += 1