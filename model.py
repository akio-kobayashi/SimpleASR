import torch
import torch.nn as nn
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
        input (B, 1, T, F) -> (B, 128, T//4, F//4)
        '''
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=64,
                            kernel_size=3, stride=1, padding=3//2, padding_mode='replicate',bias=False),
                nn.Conv2d(in_channels=64,out_channels=128,
                            kernel_size=3, stride=1, padding=3//2, padding_mode='replicate',bias=False),
                nn.BatchNorm2d(num_features=128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=128,out_channels=256,
                            kernel_size=3, stride=1, padding=3//2, padding_mode='replicate', bias=False),
                nn.Conv2d(in_channels=256,out_channels=512,
                            kernel_size=3, stride=1, padding=3//2, padding_mode='replicate', bias=False),
                nn.BatchNorm2d(num_features=512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # 176 = vocab size + 1
        self.feedforward = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 176),
                nn.Softmax()
        )

    def forward(self, x, length=None):
        x = x.unsqueeze(dim=1)
        y = self.model(x)
        if length is not None:
            mask = Len2Mask(length, y.shape[2])
            y = torch.mul(y, mask)
            y = torch.mean(torch.mean(y, dim=3), dim=2) # (B, C)
            y /= torch.sum(torch.sum(mask, dim=3), dim=2)
        else:
            y = torch.mean(torch.mean(y, dim=3), dim=2)

        y = nn.feedforward(y)

        return y
