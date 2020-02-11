import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from data import EmotionDataset
import vectorizer
import data
import rnn, cnn, mlp

models = [rnn, cnn, mlp]

from tensorboardX import SummaryWriter

def test_model(model):
    # model.load()
    print('test model: %s' % model.__name__)
    train_set, train_loader, verify_set, test_set = data.prepare(model.batch_size, model.uniform_size)
    model.train_all(train_loader, verify_set, test_set)
    model.load()
    model.judge(test_set)

from torch import nn
from torch.nn import functional as F

def dump_structure():
    # for nn, name in zip([rnn.rnn, cnn.cnn, mlp.mlp], ['rnn', 'cnn', 'mlp']):
    with SummaryWriter(comment='mlp') as w:
        # w.add_graph(mlp.mlp, (torch.randn(1, 90000),))
        # w.add_graph(rnn.rnn, (torch.randn(1, 128, 300), ))
        w.add_graph(cnn.cnn, (torch.randn(1, 306, 300), ))


        # x = torch.FloatTensor([100])
        # w.add_scalar('data/x', x, 1)

        # w.add_graph(cnn.cnn)

import warnings
warnings.filterwarnings("ignore")

import random
random.seed(0)

def test():
    for model in [mlp, rnn, cnn]:
        print('test', model.__name__)
        train_set, train_loader, verify_set, test_set = data.prepare(model.batch_size, model.uniform_size)
        model.load()
        model.judge(test_set)

def train():
    for model in [mlp, rnn, cnn]:
        train_set, train_loader, verify_set, test_set = data.prepare(model.batch_size, model.uniform_size)
        model.train_all(train_loader, verify_set, test_set)
        
    print('train over')
