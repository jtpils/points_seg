from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.utils.data as data
import numpy as np

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
    trans_tensor = torch.from_numpy(points).transpose_(1, 0)  # (N, C) -> (C, N)
    return trans_tensor.unsqueeze_(0)  # (C, N) -> (1, C, N)


class testDataSet(data.Dataset):
    def __init__(self, split='test'):
        self.split = split
        assert self.split == 'test' or self.split == 'val', print ('no such a split.')
        if self.split == 'test':
            self.pathfile = './data/test.txt'
        else:
            self.pathfile = './data/val_random_1k.txt'
        self.pathlist = []

        # read file paths
        with open(self.pathfile) as f:
            for line in f.readlines():
                self.pathlist.append(line.rstrip('\r\n'))

    def __getitem__(self, index):
        # laad data
        path = self.pathlist[index]
        points = np.load(path)

        filename = path.split('/')[-1][:-4]
        indexs = dataSplit(points, state)
        return points, indexs, filename

    def __len__(self):
        return len(self.pathlist)


class alignCollate(object):
    def __init__(self):
        super(alignCollate, self).__init__()

    def __call__(self, batch):
        points, idxes, filenames = zip(*batch)

        return points, idxes, filenames

def make_loader(split='test', batch_size=1, num_workers=0, shuffle=False, collate_fn=alignCollate()):
    return data.DataLoader(
            testDataSet(split=split),
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = shuffle,
            collate_fn = collate_fn
        )


if __name__ == '__main__':
    # pointset = testDataSet()
    # pointloader = data.DataLoader(pointset, batch_size=1, num_workers=0, shuffle=False, collate_fn=alignCollate())
    pointloader = make_loader()
    print(len(pointloader))

    for index, (points, labels, filenames) in enumerate(pointloader):

        print(len(points), points[0].shape)  # 1, (57888, 4)
        print(len(labels), len(labels[0]))  # 1, 4
        print(len(filenames), filenames[0])  # 1, 00029eea-54f4-4a60-9a15-a1a256f331b8_channelVELO_TOP

        # # left bottom
        # print(points[0][labels[0][0]].shape)

        # # right bottom
        # print(points[0][labels[0][1]].shape)

        # # left top
        # print(points[0][labels[0][2]].shape)

        # # right top
        # print(points[0][labels[0][3]].shape)

        for idx in range(4):
            print(points[0][labels[0][idx]].shape)

            # TODO: is legal
            # TODO: to tensor
            # TODO: set point_nums

            sub_points = points[0][labels[0][idx]]
            if sub_points.shape[0] != 0:
                sub_points = toTensor(sub_points)
                print(sub_points.shape)

                # TODO: to cuda
                # TODO: set num_points
                # TODO: model(sub_points)

                # sub_labels = sub_labels.detach()
                # sub_labels.squeeze_(0)
                # sub_labels = sub_labels.numpy()
                # sub_labels = np.argmax(sub_labels, axis=1)
                # pre_labels[idx] = sub_labels

        break
