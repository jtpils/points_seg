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
WORKERS = 2
OUTPUT_PATH = './results'
GPU_ID = '0'

# hyper paprams
BATCH_SIZE = 1
MODEL = 'pointnet'
NUM_CLASSES = 8
MODEL_SET = set(['pointnet'])
NUM_POINTS = 8192


# preprocess
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID




# def test_one_epoch(epoch, dataloader, model):
#     print('##### Test Epoch:%d #####' % (epoch))
#     model.eval()
#     correct_all = 0
#     for i, (points, labels) in enumerate(dataloader):
#         points = points.transpose(2, 1)
#         labels = labels.long()
#
#         points, labels = points.cuda(), labels.cuda()
#         preds = model(points)
#         preds = preds.view(-1, NUM_CLASSES)
#         labels = labels.view(-1, 1)[:, 0]
#         loss = F.nll_loss(preds, labels)
#
#         preds = preds.data.max(1)[1]
#         correct = preds.eq(labels.data).cpu().sum()
#         acc = correct.item() / float(NUM_POINTS * len(dataloader))
#         correct_all += correct
#
#         printfreq = 1
#         if i % printfreq == 0:
#             print('[%d: %d] train loss: %f accuracy: %f' % (
#             i, len(dataloader), loss.item(), correct.item() / float(BATCH_SIZE * NUM_POINTS)))
#
#     avg_acc = correct_all.item() / float(BATCH_SIZE * NUM_POINTS * len(dataloader))
#     print('avg accuracy: %f' % (acc))
#
#     return avg_acc



state = {
    'x_min': -160.0,
    'x_max': 160.0,
    'y_min': -70.0,
    'y_max': 70.0
}

def dataSplit(points, state):
    def index(x_min, x_max, y_min, y_max):
        return np.where(
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
            )

    # Outside the scope
    out_idx = np.where(
            (points[:, 0] < state['x_min']) | (points[:, 0] > state['x_max']) | (points[:, 1] < state['y_min']) | (points[:, 1] > state['y_max'])
        )

    # Inside the scope

    # left bottom
    x_min, x_max, y_min, y_max = state['x_min'], 0, state['y_min'], 0
    lb_idx = index(x_min, x_max, y_min, y_max)

    # right bottom
    x_min, x_max, y_min, y_max = 0, state['x_max'], state['y_min'], 0
    rb_idx = index(x_min, x_max, y_min, y_max)

    # left top
    x_min, x_max, y_min, y_max = state['x_min'], 0, 0, state['y_max']
    lt_idx = index(x_min, x_max, y_min, y_max)

    # right top
    x_min, x_max, y_min, y_max = 0, state['x_max'], 0, state['y_max']
    rt_idx = index(x_min, x_max, y_min, y_max)

    return (out_idx, lb_idx, rb_idx, lt_idx, rt_idx)

if __name__ == '__main__':
    # define the train/val dataloader
    print ('Loading Dataset.')
    testset = LidarDataset(split='test')
    testloader = Data.DataLoader(testset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=WORKERS)


    # load model
    model = PointNetSeg(NUM_POINTS)
    checkpoint = torch.load('./checkpoint/pointnet.pth')
    model.load_state_dict(checkpoint['model'])

    filepaths = ''
    filelist = []
    for i in xrange(len(filelist)):
        # TODO  load -> split -> 4
        # np -> tensor
        # (1, )
        # model.num
        #


    for _, (points, idxes, filename) in enumerate(testloader):
        for i in xrange(5):
            if i == 0:
                pass
            else:
                sub_points = points[0][idxes[0][i]]
                model.num_points =
                sub_lables = model(sub_points)

        # TODO  test
        # TODO  save



    # # define model
    # assert MODEL in MODEL_SET, 'no such a model.'
    #
    # model = model.cuda()
    #
    # # define optimizer
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    #
    # best_acc = 0.0
    # for epoch in range(EPOCH):
    #     adjust_learning_rate(epoch=epoch, optimizer=optimizer)
    #     train_one_epoch(epoch=epoch, dataloader=trainloader, optimizer=optimizer, model=model)
    #     acc = test_one_epoch(epoch=epoch, dataloader=valloader, model=model)
    #
    #     if acc > best_acc:
    #         best_acc = acc
    #         save_checkpoint(epoch=epoch, acc=best_acc, model=model)
    # print ('Complete.')
