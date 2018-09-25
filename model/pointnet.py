from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, num_points = 4096):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.mp1 = nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetSeg(nn.Module):
    def __init__(self, num_points = 4096, num_classes = 8):
        super(PointNetSeg, self).__init__()
        self.num_classes = num_classes
        self.num_points  = num_points
        self.stn = STN3d(num_points = self.num_points)
        self.conv1 = nn.Conv1d(3,    64,   1)
        self.conv2 = nn.Conv1d(64,   64,   1)
        self.conv3 = nn.Conv1d(64,   64,   1)
        self.conv4 = nn.Conv1d(64,   128,  1)
        self.conv5 = nn.Conv1d(128,  1024, 1)
        self.conv6 = nn.Conv1d(1088, 512,  1)
        self.conv7 = nn.Conv1d(512,  256,  1)
        self.conv8 = nn.Conv1d(256,  128,  1)
        self.conv9 = nn.Conv1d(128,  128,  1)
        self.maxpool = nn.MaxPool1d(self.num_points)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(128)
        self.cls = nn.Conv1d(128, self.num_classes, 1)



    def forward(self, x):
        # stn transform
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)

        # point feature extractor
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x

        # global feature extractor
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        globalfeat = self.maxpool(x)
        globalfeat = globalfeat.view(-1, 1024, 1).repeat(1, 1, self.num_points)

        # classfier
        x = torch.cat([pointfeat, globalfeat], 1)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))

        x = self.cls(x)
        return x

# class PointNetSeg(nn.Module):
#     def __init__(self, num_points = 4096, k = 2):
#         super(PointNetSeg, self).__init__()
#         self.num_points = num_points
#         self.k = k
#         self.feat = PointNetfeat(num_points, global_feat=False)
#         self.conv1 = torch.nn.Conv1d(1088, 512, 1)
#         self.conv2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x, trans = self.feat(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x)
#         x = x.transpose(2,1).contiguous()
#         x = F.log_softmax(x.view(-1,self.k), dim=-1)
#         x = x.view(batchsize, self.num_points, self.k)
#         return x, trans


if __name__ == '__main__':
    inputdata = Variable(torch.rand(32,3,2048))
    seg = PointNetSeg(num_points=2048)
    out= seg(inputdata)
    print('seg', out.size())