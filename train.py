
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F

from dataset.lidar import LidarDataset
from model.pointnet import PointNetSeg


# configuration
RANDOM_SEED = 10000
WORKERS = 2
OUTPUT_PATH = '.'

# hyper paprams
BATCH_SIZE = 32
NEPOCH = 30
MODEL = 'pointnet'
NUM_POINTS = 8192
NUM_CLASSES = 8
MODEL_SET = set('pointnet')

# global variables
best_acc = 0.0


def train_one_epoch(dataloader, optimizer, model):
    model.train()
    for i, (points, labels) in enumerate(dataloader):
        # (B, N, C) -> (B, C, N)
        points = points.transpose(2, 1)

        points, labels = points.cuda(), labels.cuda()
        optimizer.zero_grad()
        preds = model(points)
        loss = F.nll_loss(preds, labels)
        loss.backward()
        optimizer.step()

        # TODO: print log


def test_one_epoch(dataloader, model):
    model.eval()
    for i, (points, labels) in enumerate(dataloader):
        # (B, N, C) -> (B, C, N)
        points = points.transpose(2, 1)

        points, labels = points.cuda(), labels.cuda()
        preds = model(points)
        loss = F.nll_loss(preds, labels)

        # TODO: compute acc and iou, print log

    acc = 0.0
    if acc > best_acc:
        # TODO: save model
        pass



if __name__ == '__main__':
    # define the train/val dataloader
    trainset = LidarDataset(npoints=NUM_POINTS, split='train')
    trainloader = Data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=WORKERS)

    valset = LidarDataset(npoints=NUM_POINTS, split='val')
    valloader = Data.DataLoader(valset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=WORKERS)

    # define model
    assert MODEL in MODEL_SET, 'no such a model.'
    if MODEL == 'pointnet':
        model = PointNetSeg(num_points=NUM_POINTS, num_classes=NUM_CLASSES)
    else:
        model = PointNetSeg(num_points=NUM_POINTS, num_classes=NUM_CLASSES)

    # TODO: resume model!!!

    model = model.cuda()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_batch = len(trainset) / BATCH_SIZE

    for epoch in range(NEPOCH):
        train_one_epoch(dataloader=trainloader, optimizer=optimizer, model=model)
        test_one_epoch(dataloader=valloader, model=model)

