from spdiff import SPDiff
import argparse
import os
import yaml
from pprint import pprint
from easydict import EasyDict
import numpy as np
import random,torch
import setproctitle
torch.set_num_threads(8)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='')
    parser.add_argument('--dataset', default='')
    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
       config = yaml.safe_load(f)
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset

    config = EasyDict(config)
    agent = SPDiff(config)

    keyattr = ["lr","ft_lr", "data_dict_path", "data_dict_path", "epochs", "total_epochs", "dataset", "batch_size","diffnet", "seed"]
    keys = {}
    for k,v in config.items():
        if k in keyattr:
            keys[k] = v
    
    pprint(keys)

    agent.train()





if __name__ == '__main__':
    main()
