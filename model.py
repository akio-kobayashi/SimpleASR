# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py

def Len2Mask(length, max_len):
    short_len = [ m//4 for m in length ]
    mask = torch.zeros((len(length), 1, max_len, 1))
    for n in range(len(length)):
        mask[n, :, :length[n], :] = 1.
    return mask

class ASRModel(nn.Module):
    def __init__(self):
        super(ASRModel, self).__init__()

        '''
        input (B, 1, T, 40) -> (B, 128, T//4, 2)
        '''
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,
                      kernel_size=7, stride=1, padding=3//2,
                      padding_mode='replicate',bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (B, 32, T//2, 20)
            nn.Conv2d(in_channels=32,out_channels=64,
                      kernel_size=5, stride=1, padding=3//2,
                      padding_mode='replicate',bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (B, 64, T//4, 10)
            nn.Conv2d(in_channels=64,out_channels=128,
                      kernel_size=3, stride=1, padding=3//2,
                      padding_mode='replicate', bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # (B, 128, T//8, 5)
            nn.Conv2d(in_channels=128,out_channels=256,
                      kernel_size=3, stride=1, padding=3//2,
                      padding_mode='replicate', bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # (B, 256, T//16, 2)
        )
        # 176 = vocab size + 1
        self.fc1=nn.Linear(256*2, 176)

        self.id2word={}
        with open('vocab.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                word=line.strip().split()[0]
                id = int(line.strip().split()[1])
                self.id2word[id] = word
            
    def forward(self, x, length=None):
        # (B, C, T, F)
        x = x.unsqueeze(dim=1)
        y = self.model(x)
        if length is not None:
            mask = Len2Mask(length, y.shape[2])
            y = torch.mul(y, mask.cuda())
            y = torch.mean(y, dim=2) # (B, C, T, F)
            y /= torch.sum(mask.cuda(), dim=2)
        else:
            y = torch.mean(y, dim=2) # (B, C, T, F)
            
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        return y

    def predict(self, x):
        with torch.no_grad():
            x -= self.mean
            x /= self.std
            x = torch.from_numpy(x).clone().float()
            x = x.unsqueeze(0)
            pred = self.forward(x.cuda())
            idx = torch.argmax(pred, dim=1).to('cpu').detach().numpy().copy().astype(int)[0]
            
        return self.id2word[idx]

    def top_k(self, x, k=5):
        with torch.no_grad():
            x -= self.mean
            x /= self.std
            x = torch.from_numpy(x).clone().float()
            x = x.unsqueeze(0)
            pred = self.forward(x.cuda())
            _, idx = torch.topk(pred, k)
            idx = idx.to('cpu').detach().numpy().copy().astype(int).squeeze().tolist()
            return [ self.id2word[k] for k in idx ]
        
    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std
        
    def load_model(self, file):
        self.model.load_state_dict(torch.load(file, map_location=torch.device('cpu')), strict=False)

    def get_word(self, k):
        return self.id2word[k]
    
