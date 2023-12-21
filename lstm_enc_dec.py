import torch
import torch.nn as nn
from torch.nn.functional import embedding
import torch.optim as optim
#from torchtext.datasets import Multi30k
#from torchtext.data import Field, BucketIterator
#import spacy
import numpy as np
import random
import math
import time

import gc

device = "cuda"
#device = "cpu"

# %%
class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_token=None, emb_dim=256, hid_dim=1024, n_layers=3, dropout=0.2):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        #self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        
        src = torch.swapaxes(src, 0, 1)

        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class SingleDecoder(nn.Module):
    def __init__(self, vocab_size, pad_token=None, emb_dim=256, hid_dim=1024, n_layers=3, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        #self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        #print(input.shape)
        #print(hidden.shape)
        #print(cell.shape)

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell



class Decoder(nn.Module):
    def __init__(self, vocab_size, pad_token=None, emb_dim=256, hid_dim=1024, n_layers=3, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        #self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, enc_out):

        #print(input.shape)
        #print(hidden.shape)
        #print(cell.shape)

        hidden, cell = enc_out
        input = torch.swapaxes(input, 0, 1)

        #input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction




# %%
class Decoder2(nn.Module):
    def __init__(self, **dec_args):
        super().__init__()

        self.decoder = SingleDecoder(**dec_args)

    def forward(self, tgt, enc_out, teacher_forcing_ratio = 0.5):

        tgt = torch.swapaxes(tgt, 0, 1)

        hidden, cell = enc_out

        batch_size = tgt.shape[1]
        trg_len = tgt.shape[0]
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        input = tgt[0,:]


        for t in range(1, trg_len):
            #torch.swapaxes(hidden
            
            print(torch.cuda.memory_allocated(0))
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            #teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[t] if teacher_force else top1

        return outputs

