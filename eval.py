import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from generator import SpeechDataset
import generator
import numpy as np
import argparse
import speech_processing

class ASR():
    def __init__():
        with h5py.File('./stats.h5', 'r') as f:
            self.mean = f['mean'][()]
            self.std = f['std'][()]

            use_cuda = torch.cuda.is_available()
            torch.manual_seed(7)
            self.device = torch.device("cuda" if use_cuda else "cpu")
            if use_cuda is True:
                print('use GPU')

            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

            self.model=ASRModel()
            model.load_state_dict(torch.load('./model', map_location=torch.device('cpu')), strict=False)
            self.model.to(self.device)

            self.id2word={}
            with open('vocab.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    word = line.strip().split()[0]
                    id = int(line.strip().split()[1])
                    self.id2word[id] = word
                    
    def decode(self, wavfile):
        mel = speech_processing.compute_melspec(wavfile)
        mel = mel.unsqueeze(dim=0)

        pred = self.model(mel)
        pred = np.argmax(pred.to('cpu').detach().numpy().copy())[0]
        word = self.id2word[pred]

        return word
    
