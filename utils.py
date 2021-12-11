import os
from argparse import ArgumentParser
import scipy.io
import random
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import shutil
from config import parsers
from skimage.transform import resize

args = parsers()

def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

class TraceDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, train=True): #第一步初始化各个变量
        self.root = root
        self.train = train
        self.length = len([name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))])-1
    def __getitem__(self, idx):
        self.trace = scipy.io.loadmat(self.root+'/{}.mat'.format(idx+1))['mat']
        self.trace = self.trace[1:, 1:]
        # self.trace = resize(self.trace, (128, 128))
        # self.trace = (self.trace-0.5)*2
        label = 0
        return self.trace[None, :, :]

    def __len__(self):
        return int(self.length)


def load_data(batch_size=64, root="../images"):
    train_set = TraceDataset(root=root)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader
