import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.transpose(1,2)    # B, C, N -> B, N, C
            input = input.contiguous().view(-1,input.size(2))      # B, N, C -> B*N, C

        target = target.view(-1,1)   # B, N, 1 -> B*N, 1

        # check
        #logpt = F.log_softmax(input)
        logpt = input
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


# need to debug !!!
if __name__ == '__main__':
    input = torch.FloatTensor([[3, 1,10], [0.4, 0.1, 0.5]])
    label = torch.LongTensor([2, 2])
    loss = FocalLoss()
    print (loss(input, label))