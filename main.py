import argparse
import os
import time
import sys
import numpy as np
import models
import shutil
import utils
from utils import AverageMeter
import torchvision.transforms as transforms
import torchvision
import datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import tqdm

parser = argparse.ArgumentParser( description='Drugs Binding Affinity Prediction' )

parser.add_argument( '--save-folder', type=str, metavar='PATH' )
parser.add_argument( '--lr', type=float, default=0.001, help='initial learning rate' )
parser.add_argument( '--dataset', type=str, choices=['davis', 'kiba', 'cross'] )
parser.add_argument( '--workers', type=int, default=4, help='number of data loading workers (default:4)' )
parser.add_argument( '-b', '--batch-size', type=int, default=128, help='mini-batch size' )
parser.add_argument( '--resume', type=str, metavar='PATH' )
#parser.add_argument( '--weight-decay', type=float, default=1e-4, help='weight decay' )
parser.add_argument( '--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument( '--evaluation',dest='evaluation',action='store_true' )
parser.add_argument( '--tot-epoch', type=int, default=100 )
parser.add_argument( '--seed', type=int, default=0 )
parser.add_argument( '--protein-len', type=int, default=1200 )
parser.add_argument( '--compound-len', type=int, default=100 )
parser.add_argument( '--embedding-dim', type=int, default=128 )

class Env():
    def __init__(self, args):
        self.best_acc = 0
        self.args = args

        torch.manual_seed(0)

        logger = utils.setup_logger( os.path.join( args.save_folder, 'log.log' ) )
        self.logger = logger

        for key, value in sorted( vars(args).items() ):
            logger.info( str(key) + ': ' + str(value) )

        self.load_dataset()

        model = models.AffinityPredictNet( args.protein_len, args.compound_len, args.embedding_dim )
        self.model = model

        logger.info( 'Dims: {}'.format( sum([m.data.nelement() if m.requires_grad else 0
            for m in model.parameters()] ) ) )

        self.optimizer = optim.Adam( model.parameters(), lr=args.lr )

        self.epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info( '=> loading checkpoint from {}'.format(args.resume) )
                checkpoint = torch.load( args.resume )
                self.epoch = checkpoint['epoch']
                self.model.load_state_dict( checkpoint['model'] )
                self.optimizer.load_state_dict( checkpoint['optimizer'] )
                logger.info( '=> loaded checkpoint from {} (epoch {})'.format(
                    args.resume, self.epoch ) )
            else:
                raise Exception("No checkpoint found. Check your resume path.")

        trainDataset, testDataset = self.trainDataset, self.testDataset
        self.criterion = nn.MSELoss()

        self.trainLoader = data.DataLoader( trainDataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True )
        self.testLoader = data.DataLoader( testDataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True )

        self.args = args
        self.save( self.best_acc )
        
        self.start_time = time.time()
        self.rest_iter = (args.tot_epoch - self.epoch) * len(self.trainLoader)

        if args.evaluation:
            self.test()
        else:
            for i in range(args.tot_epoch):
                self.epoch = i+1
                self.train( i+1 )
                self.test()

    def save( self, acc ):
        logger = self.logger
        is_best = acc > self.best_acc
        self.best_acc = max( self.best_acc, acc )
        logger.info( '=> saving checkpoint' )
        utils.save_checkpoint({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best, self.args.save_folder)

    def train( self, epoch ):
        logger = self.logger
        losses = AverageMeter()

        self.model.train()
        logger.info("Training Epoch {}".format(epoch) )

        for i, batch in enumerate(self.trainLoader):
            self.optimizer.zero_grad()

            proteins, compounds, affinityScores = batch

            pred = self.model( proteins, compounds )

            loss = self.criterion( pred, affinityScores )

            losses.update( loss.item(), proteins.size(0) )
            loss.backward()
            if np.isnan( loss.item() ):
                raise Exception("Training model diverges.")
            self.optimizer.step()

            if i % self.args.print_freq == 0 and i > 0:
                log_str = 'TRAIN -> Epoch{epoch}: \tIter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})'.format( epoch=epoch, iter=i, loss=losses )
                self.logger.info( log_str )
            if i % 10 == 0 and i > 0:
                self.it = (epoch - 1) * len(self.trainLoader) + i
                finish_time = (time.time() - self.start_time) / (self.it / self.rest_iter) + self.start_time
                log_str = 'Expecting finishing time {}'.format( time.asctime( time.localtime(finish_time) ) )
                self.logger.info( log_str )

    def test( self ):
        logger = self.logger
        self.model.eval()

        MSE = AverageMeter()

        globalScores = []
        globalPred = []

        with torch.no_grad():
            for i, batch in tqdm.tqdm(enumerate(self.testLoader)):
                proteins, compounds, affinityScores = batch

                pred = self.model( proteins, compounds )
                loss = self.criterion( pred, affinityScores )
                MSE.update( loss.item(), proteins.size(0) )

                globalScores.append( affinityScores )
                globalPred.append( pred )

        globalScores = torch.cat( globalScores, 0 )
        globalPred = torch.cat( globalPred, 0 )
        cIndex = utils.ComputeCIndex( globalScores, globalPred )
        log_str = "VAL FINAL -> MSE: {}, CIndex: {}".format( MSE.avg, cIndex )
        logger.info( log_str )
        self.save( cIndex )

    def load_dataset( self ):
        args = self.args
        if args.dataset in ['davis', 'kiba']:
            trainDataset = datasets.BindingAffinityDataset( args.dataset, 'train', maxProteinLen = args.protein_len, maxCompoundLen=args.compound_len )
            testDataset = datasets.BindingAffinityDataset( args.dataset, 'test', maxProteinLen = args.protein_len, maxCompoundLen=args.compound_len )
        else:
            raise NotImplementedError('Dataset has not been implemented')
        self.trainDataset, self.testDataset = trainDataset, testDataset

if __name__ == '__main__':
    args = parser.parse_args()
    Env( args )
