#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from torch.autograd import Variable
import torch


class nwBatcher:
    
    def __init__(self, X, batch_size, num_of_samples, max_pad):
        self.X = X
        self.num_of_samples = num_of_samples
        self.batch_size = batch_size
        self.max_batch_num = int(self.num_of_samples / self.batch_size)
        self.batch_num = 0
        self.max_pad = max_pad
        

    # Get the batch with complete seqeunce
    def next_whole_seq(self):

        X_context = self.X[self.batch_num * self.batch_size : (self.batch_num + 1) * self.batch_size]
        X_target = []
        
        for i in range(len(X_context)):
            if len(X_context[i]) > self.max_pad:
                X_context[i] = X_context[i][: self.max_pad]
            
            else:
                zero_padding = [0] * max(0, self.max_pad - len(X_context[i]))
                X_context[i].extend(zero_padding)
            
            X_target.extend(X_context[i][1:] + [0])
            
        self.batch_num = (self.batch_num + 1) % self.max_batch_num
        return Variable(torch.LongTensor(X_context).cuda()), Variable(torch.LongTensor(X_target).cuda())


            
    def shuffle(self):
        np.random.shuffle(self.X)
