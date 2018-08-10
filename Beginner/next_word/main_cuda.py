#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from model_cuda import * ##change to no_cuda
from batcher_next_word import *
import cPickle as pickle
import torch
import torch.optim.lr_scheduler as s
import numpy as np
import gc
import argparse
import math

def partition(data):
    data_len = len(data)
    a = int(math.ceil(0.2 * data_len))
    b = int(math.ceil(0.9 * data_len))
    return data[:a], data[a:b], data[b:]

       
def train(train_batcher, model, criterion, optimiser):
    train_batcher.shuffle()
    step_par_epoch = train_batcher.max_batch_num
    t_total = 0
    for j in range(step_par_epoch):

        context_data, context_target = train_batcher.next_whole_seq()    
        
        
        y_pred_batch = model(context_data)
        y_pred_batch = y_pred_batch.reshape(y_pred_batch.shape[0] * y_pred_batch.shape[1], -1)
        loss = criterion(y_pred_batch, context_target)
        t_total += loss.data.item()
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        gc.collect()
        
    return str(t_total / step_par_epoch)

def evaluate(dev_batcher, model, criterion, scheduler, eval_best, output_path, description):
    total = 0
    dev_batcher.shuffle()
    step_par_epoch = dev_batcher.max_batch_num
    for j in range(step_par_epoch):
        
        context_data, context_target = dev_batcher.next_whole_seq()

        y_pred_batch  = model(context_data)
        y_pred_batch = y_pred_batch.reshape(y_pred_batch.shape[0] * y_pred_batch.shape[1], -1)
        loss = criterion(y_pred_batch, context_target)
        total += loss.data.item()

    if not eval_best or total < eval_best:
        with open(output_path + "pass_act.model_" + description, 'w') as f:
            torch.save(model,f)
            eval_best = total
    else:
        scheduler.step()   
    return eval_best, str(total / step_par_epoch)

def main(args):
    
    data = pickle.load(open(args.data_path))
    id2vec = pickle.load(open("./preprocessed_data/index_to_vector.p"))
    
    dev, train, test = partition(data)
    batch_size = args.batch_size 
    emb_dim = args.emb_size
    hidden_size = int(args.h_size)
    
    kwargs = {}
    kwargs['dropout'] = args.dropout
    kwargs['tied'] = args.tied
    kwargs['num_layers'] = 2
    kwargs['batch_size'] = batch_size
    kwargs['vocab_size'] = len(id2vec)
    kwargs['hidden_size'] = hidden_size
    kwargs['emb_dim'] = emb_dim

    description = args.description
    
    # embeddings in numpy form
    id2vec_params = []
    for i in range(max(id2vec.keys()) + 1):
        try:
            id2vec_params.append(id2vec[i])
        except:
            print 'err in id2vec'
    
    
    assert len(id2vec) == len(id2vec_params), 'id2vec_param len wrong'
    
    id2vec_params = np.array(id2vec_params)
    
    
    if args.model_name == 'lstm_nw':
        model = LSTMNextWordModel(kwargs).cuda()
        model.initGloveEmb(id2vec_params)
        train_batcher = nwBatcher(train, batch_size, len(train), args.max_pad)
        dev_batcher = nwBatcher(dev, batch_size, len(dev), args.max_pad)
        test_batcher = nwBatcher(test, batch_size, len(test), args.max_pad)
    else:
        raise ValueError('No other choices beside lstm yet')
    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimiser = torch.optim.SGD(model.parameters(), lr=20)
    scheduler = s.StepLR(optimiser, step_size=1, gamma=0.5)
    logFile = open(args.output_path + description + '.txt', 'w')
    eval_best = None    
    
    for i in range(int(args.max_epoch)):
        
        # Train
        loss = train(train_batcher, model, criterion, optimiser)
        logFile.write('train loss:' + loss +'\n')
        print('Train: ', i, ' Loss: ', loss)
    
        # Dev
        eval_best, total = evaluate(dev_batcher, model, criterion, scheduler, eval_best, args.output_path, description)
        logFile.write('eval loss:' + total + '\n')
        print('Eval: ', i, ' Loss: ', total)

    logFile.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model_name", choices=["lstm_nw"])
    parser.add_argument("--dropout", type=float, help="dropout", default=0.5)
    parser.add_argument("--max_epoch", help="max_epoch")
    parser.add_argument("--max_pad", type=int, help="max padding", default=15)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=20)
    parser.add_argument("--h_size", type=int, help="hidden_size", default=100)
    parser.add_argument("--emb_size", type=int, help="embedding_size", default=100)
    parser.add_argument("--data_path", help="path to data. No need for label", default='./preprocessed_data/wiki_nell_X.p')
    parser.add_argument("--output_path", help="output path of best model and logfile", default='./result_nw/')
    parser.add_argument("--tied", help="tied", action='store_true')
    parser.add_argument("--description", default='baseline')
    args = parser.parse_args()
    
    
    main(args)
    
