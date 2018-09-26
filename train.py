
from __future__ import print_function
from __future__ import absolute_import

import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from dataset.lidar import LidarDataset
from model.pointnet import PointNetSeg
import utils


# configuration
RANDOM_SEED = 10000
WORKERS = 2
OUTPUT_PATH = './checkpoints'
GPU_ID = '0'

# hyper paprams
BATCH_SIZE = 32
NEPOCH = 30
MODEL = 'pointnet'
NUM_POINTS = 8192
NUM_CLASSES = 8
MODEL_SET = set('pointnet')


# preprocess
cudnn.benchmark = True
torch.manual_seed(RANDOM_SEED) # cpu
torch.cuda.manual_seed(RANDOM_SEED) #gpu
np.random.seed(RANDOM_SEED) #numpy
random.seed(7) #rand
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID



def train_one_epoch(dataloader, optimizer, model):
    model.train()
    for i, (points, labels) in enumerate(dataloader):
        # (B, N, C) -> (B, C, N)
        points = points.transpose(2, 1)
        # lable: FloatTensor -> LongTensor
        labels = labels.long()

        points, labels = points.cuda(), labels.cuda()
        optimizer.zero_grad()
        preds = model(points)
        preds = preds.view(-1, NUM_CLASSES)
        labels = labels.view(-1, 1)[:, 0]
        loss = F.nll_loss(preds, labels)
        loss.backward()
        optimizer.step()

        preds = preds.data.max(1)[1]
        correct = preds.eq(labels.data).cpu().sum()

        printfreq = 1
        if i % printfreq == 0:
            print('[%d: %d] train loss: %f accuracy: %f' % (
            i, len(dataloader), loss.data[0], correct / float(BATCH_SIZE * NUM_POINTS)))


def test_one_epoch(dataloader, model):
    model.eval()
    correct_all = 0
    count_all = 0
    for i, (points, labels) in enumerate(dataloader):
        points = points.transpose(2, 1)
        labels = labels.long()

        points, labels = points.cuda(), labels.cuda()
        preds = model(points)
        preds = preds.view(-1, NUM_CLASSES)
        labels = labels.view(-1, 1)[:, 0]
        loss = F.nll_loss(preds, labels)

        preds = preds.data.max(1)[1]
        correct = preds.eq(labels.data).cpu().sum()
        correct_all += correct

        printfreq = 1
        if i % printfreq == 0:
            print('[%d: %d] train loss: %f accuracy: %f' % (
            i, len(dataloader), loss.data[0], correct / float(BATCH_SIZE * NUM_POINTS)))

    count_all = BATCH_SIZE * NUM_POINTS * len(dataloader)
    acc = correct_all / float(count_all)
    print('avg accuracy: %f' % (acc))

    return acc


def save_checkpoint(epoch, acc, model):
    print ('Saving.')
    checkpoint_path = os.path.join(OUTPUT_PATH, MODEL + '.pth')
    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['acc'] = acc
    checkpoint['model'] = model.state_dict()
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(checkpoint, checkpoint_path)

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
    model = model.cuda()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    best_acc = 0.0
    for epoch in range(NEPOCH):
        train_one_epoch(dataloader=trainloader, optimizer=optimizer, model=model)
        acc = test_one_epoch(dataloader=valloader, model=model)

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(epoch=epoch, acc=best_acc, model=model)
    print ('Complete.')
