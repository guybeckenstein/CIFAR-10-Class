import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
import torch.nn as nn
import torch.optim as optim
from utils import *
from utils import models
import utils.experiments as experiments
from utils.experiments import load_experiment
from utils.plot import plot_fit
import utils.training as training

seed: int = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()
torch.manual_seed(seed)


def plot_exp_results(filename_pattern, results_dir='results') -> None:
    fig = None
    result_files = glob.glob(os.path.join(results_dir, filename_pattern))
    result_files.sort()
    if len(result_files) == 0:
        print(f'No results found for pattern {filename_pattern}.', file=sys.stderr)
    else:
        for filepath in result_files:
            m = re.match('exp\d_(\d_)?(.*)\.json', os.path.basename(filepath))
            cfg, fit_res = load_experiment(filepath)
            fig, axes = plot_fit(fit_res, fig, legend=m[2], log_loss=True)
        del cfg['filters_per_layer']
        del cfg['layers_per_block']
        print('common config: ', cfg)


data_dir = os.path.expanduser('~/.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

''' My Hyperparameters '''
filters_per_layer: list = [64]
layers_per_block: int = 8
hidden_dims: list = [100, 100]
lr: float = 0.001
experiments.run_experiment(f'experiment_classify_cifar10',
                           seed=seed,
                           bs_train=50,
                           batches=10000,
                           epochs=30,
                           early_stopping=5,
                           reg=2e-3,
                           lr=lr,
                           filters_per_layer=filters_per_layer,
                           layers_per_block=layers_per_block,
                           pool_every=4,
                           hidden_dims=hidden_dims,
                           ycn=True)
plot_exp_results('experiment_classify_cifar10*.json')
