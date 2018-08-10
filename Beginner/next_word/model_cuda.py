#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

from torch.autograd import Variable



class LSTMNextWordModel(nn.Module):
    def __init__(self,kwargs):
        super(LSTMNextWordModel, self).__init__()
        self.emb_dim = kwargs['emb_dim']
        self.vocab_size = kwargs['vocab_size']
        self.hidden_size = kwargs['hidden_size']
        self.n_layers = kwargs['num_layers']
        self.droprate = kwargs['dropout']
        self.bidir = kwargs['bidir']
        
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.drop = nn.Dropout(self.droprate)

        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size,\
                            num_layers=self.n_layers, dropout=self.droprate, batch_first=True)
        nn.init.xavier_uniform(self.linear.weight)
        self.embeddings = nn.Embedding(self.vocab_size,self.emb_dim )
        
        if kwargs['tied']:
            if self.emb_dim != self.hidden_size:
                raise ValueError('emb size not equal to hid size')
            self.linear.weight=self.embeddings.weight 
            
    def initGloveEmb(self,init_emb):
        if self.embeddings.weight.shape[1]==300:
            print('Initialize with glove Embeddings')
            self.embeddings.weight = nn.Parameter(torch.from_numpy(init_emb).float().cuda())


    def initHidden(self, N):
        
        return (Variable(torch.randn(self.n_layers, N, self.hidden_size).zero_().cuda()),
                Variable(torch.randn(self.n_layers, N, self.hidden_size).zero_().cuda()))

    def forward(self, context_input):
        context = self.drop(self.embeddings(context_input))
        init_hidd = self.initHidden(context_input.size()[0])
        context_whole_seq, _ = self.lstm(context, init_hidd)
        
        context_d = self.drop(context_whole_seq)
        context_l = self.linear(context_d)
        
        return context_l
        
        
        
