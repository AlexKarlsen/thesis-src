from branchymodels import MSDNet
from op_counter import measure_model
import argparse

if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(description='MSDNet Evaluation')
    parser.add_argument('--nBlocks', default=5)
    parser.add_argument('--step', default=4)
    parser.add_argument('--stepmode', default='even')
    parser.add_argument('--base', default=4)
    parser.add_argument('--nChannels', default=32)
    parser.add_argument('--growthRate', default=16)
    parser.add_argument('--nScales', default=4, help="lenght of grFactor")
    parser.add_argument('--grFactor',  nargs=4, default=[1, 2, 4, 4])
    parser.add_argument('--bnFactor',  nargs=4, default=[1, 2, 4, 4])
    parser.add_argument('--data', default='ImageNet')
    parser.add_argument('--prune', default='max', choices=['min', 'max'])
    parser.add_argument('--bottleneck', default=True, type=bool)
    parser.add_argument('--reduction', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')

    args = parser.parse_args()


    model = MSDNet.MSDNet(args)

    flops, parameters = measure_model(model, 224, 224)

    print('Flops = {}\nParameters = {}'.format(sum(flops), sum(parameters)))
