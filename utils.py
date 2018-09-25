'''
Some helper functions for our project.
'''

from __future__ import print_function
from __future__ import absolute_import

import torch

def index2onehot(label, num_classes):
    batch_size = label.size(0)
    labels_onehot = torch.FloatTensor(batch_size, 2, num_classes).type_as(label).zero_()
    labels_onehot.scatter_(-1, label, 1)
    return labels_onehot


if __name__ == '__main__':
    test = torch.LongTensor([[0, 2], [7, 1]])
    print (test.shape)
    print (test.shape)
    print (test)
    print (index2onehot(test, 8).shape)