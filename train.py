import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from generator import SpeechDataset
import generator
import numpy as np
import argparse

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def train(model, loader, optimizer, scheduler,
          criterion, epoch, iter_meter, writer):
    model.train()

    data_len=len(loader)

    epoch_loss = 0
    for batch_idx, _data in enumerate(loader):
        inputs, labels, lengths = _data

        optimizer.zero_grad()

        output = model(inputs.cuda(), lengths)
        loss = criterion(output, labels.to(device))
        if writer:
            writer.add_scalar('loss', loss.item(), iter_meter.get())
        loss.backward()

        optimizer.step()
        scheduler.step()
        iter_meter.step()

        output = output.view(-1, 1)
        labels = labels.view(-1, 1)
        
        if batch_idx > 0 and (batch_idx % 100 == 0 or batch_idx == data_len) :
            print('Train Epcoh: {} [{}/{} ({:.0f}%)]\t Loss: {:.9f} Acc: {:.3f}'.format(
                epoch, batch_idx * len(inputs), data_len*inputs.shape[0],
                100. * batch_idx / data_len, loss.item(), 100. * acc)
            )

        epoch_loss += loss.item()

        del loss
        torch.cuda.empty_cache()

    epoch_loss /= data_len

    print('Train Epcoh: {}\t Loss: {:.9f}'.format(epoch, epoch_loss))

    return epoch_loss

def evaluate(model, loader):

    model.eval()
    data_len = len(loader)
    preds=None
    corrs=None

    with torch.no_grad():
        for i, _data in enumerate(loader):
            inputs, labels, lengths = _data

            output = model(inputs.cuda(), lengths)

            outputs=output.to('cpu').detach().numpy().copy()
            outputs = outputs.view(-1, 1)
            labels=labels.to('cpu').detach().numpy().copy().astype(float)
            labels = labels.view(-1, 1)
            
            if preds is None:
                preds=outputs
                corrs=labels
            else:
                preds = np.append(preds, outputs)
                corrs = np.append(corrs, labels)

    return preds, corrs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data(.h5)')
    parser.add_argument('--train', type=str, required=True, help='training keys')
    parser.add_argument('--valid', type=str, required=True, help='validation keys')
    parser.add_argument('--out-stats', type=str, default='stats.h5', help='mean/std values for norm')
    parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--output', type=str, default='model', help='output model')
    parser.add_argument('--learning-rate', type=float, default=1.0e-4, help='initial learning rate')
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.logs)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda is True:
        print('use GPU')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset=SpeechDataset(path=args.data, keypath=args.train, train=True)
    mean, std = train_dataset.get_stats()

    train_loader =data.DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=lambda x: generator.data_processing(x,'train'),
                                  **kwargs)

    valid_dataset=SpeechDataset(path=args.data, keypath=args.valid, stats=[mean, std], train=False)
    valid_loader=data.DataLoader(dataset=valid_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=lambda x: generator.data_processing(x, 'valid'))

    model=ASRModel()

    print(f'The model has {count_parameters(model):,} trainable parameters')

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr = args.learning_rate,
                                betas=(0.9, 0.98),
                                eps=1.0e-9)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=args.learning_rate,
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=args.epochs,
                                              anneal_strategy='linear')
    criterion=nn.CrossEntropyLoss()
    iter_meter=IterMeter()

    max_acc=0.
    for epoch in range(1, args.epochs+1):
        train(model, train_loader,
              optimizer, scheduler, criterion,
              epoch, iter_meter, writer)
        # get predictions shape=(B, 1)
        preds, corrs = evaluate(model, valid_loader)
        print('Valid Acc: %.3f Prec: %.3f Recall: %.3f F1: %.3f' % (acc, prec, recall, f1))

        if acc > max_acc:
            print('Maximum Acc changed... %.3f -> %.3f' % (max_acc , acc))
            max_acc = acc
            torch.save(model.to('cpu').state_dict(), args.output)
            model.to(device)

if __name__ == "__main__":
    main()