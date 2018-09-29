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
from tensorboardX import SummaryWriter

from dataset.lidar import LidarDataset
from model.pointnet import PointNetSeg
import utils


# configuration
RANDOM_SEED = 10000
WORKERS = 2
OUTPUT_PATH = './checkpoints'
GPU_ID = '0'

# hyper paprams
BATCH_SIZE = 4
EPOCH = 30
MODEL = 'pointnet'
NUM_POINTS = 8192
NUM_CLASSES = 8
GAMMA = 0.2
MODEL_SET = set(['pointnet'])
SCHEDULE = set([10, 20])
LEARNING_RATE = 0.001


# preprocess
writer = SummaryWriter()
cudnn.benchmark = True
torch.manual_seed(RANDOM_SEED) # cpu
torch.cuda.manual_seed(RANDOM_SEED) #gpu
np.random.seed(RANDOM_SEED) #numpy
random.seed(RANDOM_SEED) #rand
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID



def train_one_epoch(epoch, dataloader, optimizer, model):
    print ('##### Train Epoch:%d #####' % (epoch))
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
        acc = correct.item() / float(BATCH_SIZE * NUM_POINTS)

        printfreq = 1
        if i % printfreq == 0:
            print('[%d: %d] train loss: %f accuracy: %f' % (i, len(dataloader), loss.item(), acc))

        # log vis
        writer.add_scalar('train loss', loss.item(), epoch * len(dataloader) + i)
        writer.add_scalar('train acc',  acc,         epoch * len(dataloader) + i)



def test_one_epoch(epoch, dataloader, model):
    print('##### Test Epoch:%d #####' % (epoch))
    model.eval()
    correct_all = 0
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
        acc = correct_all.item() / float(NUM_POINTS * len(dataloader))
        correct_all += correct

        printfreq = 1
        if i % printfreq == 0:
            print('[%d: %d] train loss: %f accuracy: %f' % (
            i, len(dataloader), loss.item(), correct.item() / float(BATCH_SIZE * NUM_POINTS)))

        # log vis
        writer.add_scalar('val loss', loss.item(), epoch * len(dataloader) + i)
        writer.add_scalar('val acc',  acc,         epoch * len(dataloader) + i)

    avg_acc = correct_all.item() / float(BATCH_SIZE * NUM_POINTS * len(dataloader))
    writer.add_scalar('val avg_acc', avg_acc, epoch)
    print('avg accuracy: %f' % (acc))

    return avg_acc


def save_checkpoint(epoch, acc, model):
    print ('Saving.')
    checkpoint_path = os.path.join(OUTPUT_PATH, MODEL + '.pth')
    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['acc'] = acc
    checkpoint['model'] = model.state_dict()
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    torch.save(checkpoint, checkpoint_path)

def adjust_learning_rate(epoch, optimizer):
    global LEARNING_RATE
    if epoch in SCHEDULE:
        LEARNING_RATE *= GAMMA
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE

if __name__ == '__main__':
    # define the train/val dataloader
    print ('Loading Dataset.')
    trainset = LidarDataset(npoints=NUM_POINTS, split='train')
    trainloader = Data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=WORKERS)

    valset = LidarDataset(npoints=NUM_POINTS, split='val')
    valloader = Data.DataLoader(valset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=WORKERS)

    # define model
    assert MODEL in MODEL_SET, 'no such a model.'
    if MODEL == 'pointnet':
        print('Loading PointNet.')
        model = PointNetSeg(num_points=NUM_POINTS, num_classes=NUM_CLASSES)
    # default model is pointnet.
    else:
        print('Loading PointNet.')
        model = PointNetSeg(num_points=NUM_POINTS, num_classes=NUM_CLASSES)
    model = model.cuda()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    best_acc = 0.0
    for epoch in range(EPOCH):
        adjust_learning_rate(epoch=epoch, optimizer=optimizer)
        train_one_epoch(epoch=epoch, dataloader=trainloader, optimizer=optimizer, model=model)
        acc = test_one_epoch(epoch=epoch, dataloader=valloader, model=model)

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(epoch=epoch, acc=best_acc, model=model)
    print ('Complete.')
