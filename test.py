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

    return (lb_idx, rb_idx, lt_idx, rt_idx)


def toTensor(points):
    trans_tensor = torch.from_numpy(points).transpose_(1, 0)
    return trans_tensor.unsqueeze_(0)

def testLoader(test_set_file):
    with open(test_set_file, 'r') as ptr:
        for test_file in ptr:
            filename = test_file.strip().split('/')[-1].split('.')[0]
            points = np.load(test_file[:-2])
            points = points[0:4]
            lb_idx, rb_idx, lt_idx, rt_idx = dataSplit(points, state)
            yield (filename, points, (lb_idx, rb_idx, lt_idx, rt_idx))



if __name__ == '__main__':
    # load model
    model = PointNetSeg(NUM_POINTS)
    checkpoint = torch.load('./checkpoints/pointnet.pth')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()


    #filepaths = './data/test.txt'
    filepaths = './data/val_random_1k.txt'
    out_dir = OUTPUT_PATH_ROOT
    COUNT = 0
    for filename, points, idxes in testLoader(filepaths):
        print('##### Processing Points: %d #####' % (COUNT))
        COUNT += 1
        pre_labels = np.zeros((points.shape[0], ))
        for idx in idxes:
            sub_points = toTensor(points[idx])
            sub_points = sub_points.cuda()
            model.set_num_points(sub_points.size(2))

            sub_labels = model(sub_points)
            sub_labels = sub_labels.detach()
            sub_labels.squeeze_(0)
            sub_labels = sub_labels.numpy()
            sub_labels = np.argmax(sub_labels, axis=1)
            pre_labels[idx] = sub_labels

        np.savetxt(
            os.path.join(out_dir, filename + '.csv'),
            pre_labels,
            fmt='%d',
            delimiter=','
        )