from SELFRec import SELFRec
from util.conf import ModelConf
import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # Register your model here
    graph_baselines = ['LightGCN', 'DirectAU', 'MF']
    ssl_graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF']
    sequential_baselines = ['SASRec']
    ssl_sequential_models = ['CL4SRec', 'DuoRec', 'BERT4Rec']

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)

    print('Graph-Based Baseline Models:')
    print('   '.join(graph_baselines))
    print('-' * 100)
    print('Self-Supervised  Graph-Based Models:')
    print('   '.join(ssl_graph_models))
    print('=' * 80)
    print('Sequential Baseline Models:')
    print('   '.join(sequential_baselines))
    print('-' * 100)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('=' * 80)
    # chosen_baselines = ['LightGCN', 'MF', 'DirectAU', 'MixGCF', 'SGL', 'SimGCL', 'NCL']
    chosen_baselines = ['NCL']
    # chosen_baselines = ['NCL']
    dataset = ["google"]

    for dataset in dataset:
        for model in chosen_baselines:
            # model = input('Please enter the model you want to run:')
            import time
            setup_seed(2025)
            s = time.time()
            if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
                conf = ModelConf('./conf/' + model + '.conf')
                conf.change('training.set', f"../src/data/{dataset}/train.txt")
                conf.change('test.set', f"../src/data/{dataset}/test.txt")
                # conf.change('num.max.epoch', "1")
                conf.add('dataset', dataset)
            else:
                print('Wrong model name!')
                exit(-1)
            rec = SELFRec(conf)
            rec.execute()
            e = time.time()
            print("Running time: %f s" % (e - s))
