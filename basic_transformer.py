
import math

import torch
import torch.nn as nn



device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

class PositionalEncoding(nn.Module):
    # Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    # Following pytorch tutorial
    def __init__(self, pad_token, vocab_size, d_model=512, nhead=8, nlayer=6, d_hid=2048, dropout=0.2):
        super().__init__()

        self.pad_token = pad_token
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, nlayer)

        #self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src):

        pad_mask = src == self.pad_token # * float("-inf")


        #print(torch.all(torch.isnan(src) == False))

        src = self.embedding(src) #* math.sqrt(self.d_model)   # Why this norm thing???
        src = self.pos_enc(src)

        
        # Look more into proper masking later
        # Mask pad tokens?

        output = self.encoder(src, src_key_padding_mask=pad_mask)
        #output = self.encoder(src)
        

        return output



class Decoder(nn.Module):
    def __init__(self, pad_token, vocab_size, d_model=512, nhead=8, nlayer=6, d_hid=2048, dropout=0.2):
        super().__init__()

        self.d_model = d_model
        self.pad_token = pad_token

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, nlayer)

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):

        pad_mask = tgt == self.pad_token #* float("-inf")
        
        
        tgt = self.embedding(tgt) #* math.sqrt(self.d_model)   # Why this norm thing??? 

        tgt = self.pos_enc(tgt)

    
        if tgt_mask is None:
            # Look more into proper masking later
            # Could mask encoder pad tokens?
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
            tgt_mask = (tgt_mask != 0).to(device)

            #tgt_mask = torch.triu(torch.ones((tgt.shape[1], tgt.shape[1])), diagonal=1).to(device)


            #tgt_mask = pad_mask and tgt_mask
            # Combine with input tgt mask


        output = self.decoder(tgt, enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask)
        output = self.linear(output)
 
        return output






