import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def Len2Mask(length, max_len):
    short_len = [ m//4 for m in length ]
    mask = torch.zeros((len(length), 1, max_len, 1))
    for n in range(len(length)):
        mask[n, :, :length[n], :] = 1.
    return mask

class ASRModel(nn.Module):
    def __init__(self, ):
        super(ASRModel, self).__init__()

        '''
        input (B, 1, T, 40) -> (B, 128, T//4, 2)
        '''
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,
                      kernel_size=7, stride=1, padding=3//2,
                      padding_mode='replicate',bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (B, 32, T//2, 20)
            nn.Conv2d(in_channels=32,out_channels=64,
                      kernel_size=5, stride=1, padding=3//2,
                      padding_mode='replicate',bias=False),
            nn.BatchNorm2d(num_features=64),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (B, 64, T//4, 10)
            nn.Conv2d(in_channels=64,out_channels=128,
                      kernel_size=3, stride=1, padding=3//2,
                      padding_mode='replicate', bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (B, 128, T//8, 5)
            nn.Conv2d(in_channels=128,out_channels=256,
                      kernel_size=3, stride=1, padding=3//2,
                      padding_mode='replicate', bias=False),
            nn.BatchNorm2d(num_features=256),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # (B, 256, T//16, 2)
        )
        # 176 = vocab size + 1
        self.fc1=nn.Linear(256*2, 176)
        
    def forward(self, x, length=None):
        x = x.unsqueeze(dim=1)
        y = self.model(x)
        if length is not None:
            mask = Len2Mask(length, y.shape[2])
            y = torch.mul(y, mask.cuda())
            y = torch.mean(y, dim=2) # (B, C, 2)
            y /= torch.sum(mask.cuda(), dim=2)
        else:
            y = torch.mean(y, dim=2) # (B, C, 2)
            
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        #y = F.softmax(y, dim=1)
        return y
