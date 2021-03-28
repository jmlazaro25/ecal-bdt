#!/usr/bin/python

import os
import sys
import math
import random
import logging
import argparse
import importlib
import ROOT as r
import numpy as np
import pickle as pkl
import xgboost as xgb
import matplotlib as plt

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

plt.use('Agg')
from collections import Counter
from array import array
from optparse import OptionParser
sys.path.insert(0, '../')
from sklearn import metrics

class sampleContainer:
    def __init__(self,filename,maxEvts,isSig):
        print("Initializing Container!")
        self.tree = r.TChain("HcalVeto")
        self.tree.Add(filename)

        self.maxEvts   = maxEvts
        self.isSig = isSig

    def root2PyEvents(self):
        self.events =  []
        for event in self.tree:
            if len(self.events) >= self.maxEvts:
                continue
            evt = []

            ###################################
            # Features
            ###################################
            evt.append(event.nHits)      # 0
            evt.append(event.centE)      # 1
            evt.append(event.centPE)     # 2
            evt.append(event.std_e_z)    # 3
            evt.append(event.totE)       # 4
            evt.append(event.totPE)      # 5
            evt.append(event.maxE)       # 6
            evt.append(event.maxPE)      # 7
            evt.append(event.dz_e)       # 8

            evt.append(event.std_e_1_r)  # 9
            evt.append(event.std_e_1_e)  # 10
            evt.append(event.std_e_1_pe) # 11
            evt.append(event.avg_e_1_e)  # 12
            evt.append(event.avg_e_1_pe) # 13
            evt.append(event.tot_e_1_e)  # 14
            evt.append(event.tot_e_1_pe) # 15
            evt.append(event.max_e_1_e)  # 16
            evt.append(event.max_e_1_pe) # 17

            evt.append(event.std_e_2_r)  # 18
            evt.append(event.std_e_2_e)  # 19
            evt.append(event.std_e_2_pe) # 20
            evt.append(event.avg_e_2_e)  # 21
            evt.append(event.avg_e_2_pe) # 22
            evt.append(event.tot_e_2_e)  # 23
            evt.append(event.tot_e_2_pe) # 24
            evt.append(event.max_e_2_e)  # 25
            evt.append(event.max_e_2_pe) # 26

            evt.append(event.std_e_3_r)  # 27
            evt.append(event.std_e_3_e)  # 28
            evt.append(event.std_e_3_pe) # 29
            evt.append(event.avg_e_3_e)  # 30
            evt.append(event.avg_e_3_pe) # 31
            evt.append(event.tot_e_3_e)  # 32
            evt.append(event.tot_e_3_pe) # 33
            evt.append(event.max_e_3_e)  # 34
            evt.append(event.max_e_3_pe) # 35

            self.events.append(evt)

        new_idx=np.random.permutation(np.arange(np.shape(self.events)[0]))
        self.events = np.array(self.events)
        np.take(self.events, new_idx, axis=0, out=self.events)
        print("Final Event Shape" + str(np.shape(self.events)))
        #self.tfile.Close()

    def constructTrainAndTest(self):
        self.train_x = self.events
        self.train_y = np.zeros(len(self.train_x)) + (self.isSig == True)

class mergedContainer:
    def __init__(self, sigContainer,bkgContainer):
        self.train_x = np.vstack((sigContainer.train_x,bkgContainer.train_x))
        self.train_y = np.append(sigContainer.train_y,bkgContainer.train_y)
        
        self.train_x[np.isnan(self.train_x)] = 0.000
        self.train_y[np.isnan(self.train_y)] = 0.000
        
        self.dtrain = xgb.DMatrix(self.train_x,self.train_y)
    

if __name__ == "__main__":
    
    parser = OptionParser()


    parser.add_option('--seed', dest='seed',type="int",  default=2, help='Numpy random seed.')
    parser.add_option('--max_evt', dest='max_evt',type="int",  default=1500000, help='Max Events to load')
    parser.add_option('--out_name', dest='out_name',  default='test', help='Output Pickle Name')
    parser.add_option('--eta', dest='eta',type="float",  default=0.023, help='Learning Rate')
    parser.add_option('--tree_number', dest='tree_number',type="int",  default=1000, help='Tree Number')
    parser.add_option('--depth', dest='depth',type="int",  default=10, help='Max Tree Depth')
    parser.add_option('--bkg_file', dest='bkg_file', default='bdt_0/bkg_train.root', help='name of background file')
    parser.add_option('--sig_file', dest='sig_file', default='bdt_0/sig_train.root', help='name of signal file')
  

    (options, args) = parser.parse_args()

    np.random.seed(options.seed)
    
    adds=0
    Check=True
    while Check:
        if not os.path.exists(options.out_name+'_'+str(adds)):
            try:
                os.makedirs(options.out_name+'_'+str(adds))
                Check=False
            except:
               Check=True
        else:
            adds+=1


    print("Random seed is = %s" % (options.seed))
    print("You set max_evt = %s" % (options.max_evt))
    print("You set tree number = %s" % (options.tree_number))
    print("You set max tree depth = %s" % (options.depth))
    print("You set eta = %s" % (options.eta))

    print('Loading sig_file = %s' % (options.sig_file))
    sigContainer = sampleContainer(options.sig_file,options.max_evt,True)
    sigContainer.root2PyEvents()
    sigContainer.constructTrainAndTest()

    print('Loading bkg_file = %s' % (options.bkg_file))
    bkgContainer = sampleContainer(options.bkg_file,options.max_evt,False)
    bkgContainer.root2PyEvents()
    bkgContainer.constructTrainAndTest()

    eventContainer = mergedContainer(sigContainer,bkgContainer)

    params     = {"objective": "binary:logistic",
                "eta": options.eta,
                "max_depth": options.depth,
                "min_child_weigrt": 20,
                "silent": 1,
                "subsample": .9,
                "colsample_bytree": .85,
                #"eval_metric": 'auc',
                "eval_metric": 'error',
                "seed": 1,
                "nthread": 1,
                "verbosity": 1,
                "early_stopping_rounds" : 10}

    num_trees  = options.tree_number
    gbm       = xgb.train(params, eventContainer.dtrain, num_trees)

    output = open(options.out_name+'_'+str(adds)+"/"+options.out_name+'_'+str(adds)+"_weights"+'.pkl', 'wb')
    pkl.dump(gbm, output)

    xgb.plot_importance(gbm)
    plt.pyplot.savefig(options.out_name+'_'+str(adds)+"/"+options.out_name+'_'+str(adds)+'_fimportance.png', dpi=500, bbox_inches='tight', pad_inches=0.5)

print("Files saved in: ", options.out_name+'_'+str(adds))
