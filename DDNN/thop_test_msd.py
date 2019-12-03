import torch
from branchymodels.MSDNet import MSDNet
from thop import profile
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
    model = MSDNet(args)


    x = torch.randn(1, 3, 224, 224)

    flops = []
    params = []

    fall, pall = profile(model, inputs=(x, ))
    #flops.append(f)
    #params.append(p)

    for i, (b, c) in enumerate(zip(model.blocks, model.classifier)):
        f, p = profile(b, inputs=(x, ))
        flops.append(f)
        params.append(p)
        x = b(x)
        f, p = profile(c, inputs=(x, ))
        flops[i] += f
        params[i] += p


    print(flops, params)
    print(sum(flops), sum(params))
    print(fall, pall)
