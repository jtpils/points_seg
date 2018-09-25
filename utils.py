'''
Some helper functions for our project.
'''

from __future__ import print_function
from __future__ import absolute_import

import torch

def index2onehot():
    pass


if __name__ == '__main__':
    test = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]])
    print (test)