import os
import json
import math
from tqdm import tqdm

import torch
import torchnet as tnt

import sys
sys.path.append("/home/jason/pytorch-inceptionv3")

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils

from data_utils.pytorch_birds_data_loader import PytorchBirdsDataLoader

def main(opt):
    # load model
    #model = torch.load(opt['model.model_path'])
    print('opt:', opt)
    model = model_utils.load(opt)
    model.eval()

    # load opts
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # Postprocess arguments
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')

    # construct data
    data_opt = { 'data.' + k: v for k,v in filter_opt(model_opt, 'data').items() }

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k,v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    # added for birds dataset
    data_opt['data.test_way']      = 5
    data_opt['data.test_shot']     = 5
    data_opt['data.test_query']    = (41 - 5) # max amount possible
    data_opt['data.test_episodes'] = 1000 # average over 1000 randomly generated episodes

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    #data = data_utils.load(data_opt, ['test'])
    test_way = data_opt['data.test_way']
    test_shot = data_opt['data.test_shot']
    test_query = data_opt['data.test_query']
    test_episodes = data_opt['data.test_episodes']
    print('test way:', test_way, 'test shot:', test_shot, 'test query:', test_query, 'test_episodes', test_episodes)
    # figure out how to go through all the data with 5 shot
    data = PytorchBirdsDataLoader(
        n_episodes=test_episodes,
        n_way=test_way,
        n_query=test_query,
        n_support=test_shot
    )

    if data_opt['data.cuda']:
        model.cuda()

    meters = { field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields'] }

    model_utils.evaluate(model, data, meters, desc="test")

    for field,meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(data_opt['data.test_episodes'])))
