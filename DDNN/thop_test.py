import torch
from conventionalmodels.resnet import ResNet
from conventionalmodels.densenet import DenseNet
from branchymodels.BResNet import BResNet
from branchymodels.BDenseNet import BDenseNet
from branchymodels.MSDNet import MSDNet
from thop import profile
import argparse

def measure_model_all_exits(args, model=None):
    if model == 'resnet':
        model = ResNet(100)
    elif model == 'densenet':
        model = DenseNet(100)
    elif model == 'bresnet':
        model = BResNet(100)
    elif model == 'bdensenet':
        model = BDenseNet(100)
    elif model == 'msdnet':
        model = MSDNet(args)
    else:
        pass
    
    models = {'resnet' : ResNet(100), 'densenet' : DenseNet(100), 'bdensenet' : BDenseNet(100), 'bdensenet' : BDenseNet(100), 'msdnet' : MSDNet(args)}
    flops = []
    params = []
    x= torch.randn(1, 3, 224, 224)

    stats = {x: {} for x in models}

    for model in models:
        f, p = profile(models[model], x=(x))
        stats[model]['flops'] = f
        stats[model]['params'] = p

    return stats

def measure_bresnet_exits():
    flops = []
    params = []
    model = BResNet(100)
    x = torch.randn(1, 3, 224, 224)
    fall, pall = profile(model, inputs=(x,))


    
    f, p = profile(model.conv1, inputs=(x, ))
    flops.append(f)
    params.append(p)

    x = model.conv1(x)

    f, p = profile(model.exit1, inputs=(x, ))
    flops[0] += f
    params[0] += p

    _, x = model.exit1(x)


    f, p = profile(model.exit2, inputs=(x, ))
    flops.append(f)
    params.append(p)

    _, x = model.exit2(x)

    f, p = profile(model.exit3, inputs=(x, ))
    flops.append(f)
    params.append(p)

    _, x = model.exit3(x)

    f, p = profile(model.exit4, inputs=(x, ))
    flops.append(f)
    params.append(p)

    print(flops, params)
    print(sum(flops), sum(params))
    print(fall, pall)

def measure_bdensenet_exits():
    flops = []
    params = []
    bdensenet = BDenseNet(100)
    x = torch.randn(1, 3, 224, 224)
    fall, pall = profile(bdensenet, inputs=(x,))

    f, p = profile(bdensenet.exit1, inputs=(x, ))
    flops.append(f)
    params.append(p)

    _, x = bdensenet.exit1(x)

    
    f, p = profile(bdensenet.transistion1, inputs=(x, ))
    flops.append(f)
    params.append(p)

    x = bdensenet.transistion1(x)

    f, p = profile(bdensenet.exit2, inputs=(x, ))
    flops[1] += f
    params[1] += p

    _, x = bdensenet.exit2(x)
    
    f, p = profile(bdensenet.transistion2, inputs=(x, ))
    flops.append(f)
    params.append(p)

    x = bdensenet.transistion2(x)

    f, p = profile(bdensenet.exit3, inputs=(x, ))
    flops[2] += f
    params[2] += p

    _, x = bdensenet.exit3(x)
    
    f, p = profile(bdensenet.transistion3, inputs=(x, ))
    flops.append(f)
    params.append(p)

    x = bdensenet.transistion3(x)

    f, p = profile(bdensenet.exit4, inputs=(x, ))
    flops[3] += f
    params[3] += p


    print(flops, params)
    print(sum(flops), sum(params))
    print(fall, pall)

    # print(sum(stats['BDenseNet']['params']))
    # print(sum(stats['BDenseNet']['flops']))
    # return {
    #     'BDenseNet' : {
    #         'params': [p for p in params],
    #         'flops': [f for f in flops]
    #     }
    # }

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
    measure_bresnet_exits()
    