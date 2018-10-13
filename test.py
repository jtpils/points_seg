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

from model.pointnet import PointNetSeg
from dataset.lidar_test import make_loader
import utils


# configuration
WORKERS = 2
OUTPUT_PATH_ROOT = './results'
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


def toTensor(points):
    trans_tensor = torch.from_numpy(points).transpose_(1, 0)  # (N, C) -> (C, N)
    return trans_tensor.unsqueeze_(0)  # (C, N) -> (1, C, N)


if __name__ == '__main__':
    # load model
    model = PointNetSeg(NUM_POINTS)
    checkpoint = torch.load('./checkpoints/pointnet.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()

    out_dir = OUTPUT_PATH_ROOT
    
    testDataLoader = make_loader(split='test', batch_size=1, num_workers=0, shuffle=False)
    COUNT = 0
    for index, (points, labels, filenames) in enumerate(testDataLoader):

        # print(len(points), points[0].shape)  # 1, (57888, 4)
        # print(len(labels), len(labels[0]))  # 1, 4
        # print(len(filenames), filenames[0])  # 1, 00029eea-54f4-4a60-9a15-a1a256f331b8_channelVELO_TOP

        print('##### Processing Points: %d #####' % (COUNT))
        COUNT += 1

        pre_labels = np.zeros((points[0].shape[0], ))
        for idx in range(4):
            sub_points = points[0][labels[0][idx]]
            if sub_points.shape[0] != 0:
                sub_points = toTensor(sub_points)   # (1, C, N)
                sub_points = sub_points.cuda()
                model.set_num_points(sub_points.size(2))
                
                sub_labels = model(sub_points)
                sub_labels = sub_labels.detach()
                sub_labels.squeeze_(0)  # (N, 8)
                sub_labels = sub_labels.cpu().numpy()
                sub_labels = np.argmax(sub_labels, axis=1)
                pre_labels[labels[0][idx]] = sub_labels

        np.savetxt(
            os.path.join(out_dir, filenames[0] + '.csv'),
            pre_labels,
            fmt='%d',
            delimiter=','
        )
