import analyze_local
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze edge offloading results')
    parser.add_argument('--filepath', default='local/nuc_local_b-resnet.json')
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--test', default='delay-threshold'),
    parser.add_argument('--model-type', default='b-resnet'),
    parser.add_argument('--weights', default=[1,1.2,1.4,1.6,1.6])
    args = parser.parse_args()

    models = ['resnet', 'densenet', 'b-resnet', 'b-densenet', 'msdnet']
    #platforms = ['gpu', 'jetson', 'nuc']

    #for platform in platforms:
    for model in models:
        analyze_local.main('gpu', model, args.test)