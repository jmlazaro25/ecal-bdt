import os
import logging

import numpy
import pickle
import xgboost
import matplotlib

import ROOT
import core

core.load_event_dict()

class SampleContainer:
    def __init__(self,filename,maxEvts,isSig):
        f = ROOT.TFile.Open(filename)
        t = f.Get('LDMX_Events')

        events = []
        events_left = maxEvts
        for event in t :
            events.append(core.translation(event))
            events_left -= 1
            if events_left < 0 :
                break

        # done with loop through event tree
        # DONT USE f or t after we close the file
        f.Close()

        new_idx=numpy.random.permutation(numpy.arange(numpy.shape(events)[0]))
        events = numpy.array(events)
        numpy.take(events, new_idx, axis=0, out=events)

        self.train_x = events
        self.train_y = numpy.zeros(len(self.train_x)) + (isSig == True)

class Trainer :
    def __init__(self, sig_sample, bkg_sample, eta, depth, tree_number) :

        train_x = numpy.vstack((sig_sample.train_x,bkg_sample.train_x))
        train_y = numpy.append(sig_sample.train_y,bkg_sample.train_y)
        
        train_x[numpy.isnan(train_x)] = 0.000
        train_y[numpy.isnan(train_y)] = 0.000
        
        self.unified_training_sample = xgboost.DMatrix(train_x,train_y)
        self.training_parameters = {'objective': 'binary:logistic',
                  'eta': eta,
                  'max_depth': depth,
                  'min_child_weigrt': 20,
                  'silent': 1,
                  'subsample':.9,
                  'colsample_bytree': .85,
                  #'eval_metric': 'auc',
                  'eval_metric': 'error',
                  'seed': 1,
                  'nthread': 1,
                  'verbosity': 1,
                  'early_stopping_rounds' : 10}
        self.tree_number = tree_number

    def train(self) :
        # Actual training
        self.gbm = xgboost.train(self.training_parameters, self.unified_training_sample, self.tree_number)

    def save(self, name, plot_importance=True) :
        # Store BDT
        with open(f'{arg.out_name}_weights.pkl','wb') as output :
            pickle.dump(self.gbm, output)
    
        # Plot feature importances
        xgboost.plot_importance(self.gbm)
        matplotlib.pyplot.savefig(f'{arg.out_name}_fimportance.png',
                dpi=500, bbox_inches='tight', pad_inches=0.5) # png parameters
        

if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse
    parser = argparse.ArgumentParser(f'ldmx python3 {sys.argv[0]}', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b',metavar='BKGD_FILE',dest='bkg_file',required=True,help='Path to background file')
    parser.add_argument('-s',metavar='SIGN_FILE',dest='sig_file',required=True,help='Path to signal file')
    parser.add_argument('-o',metavar='OUT_NAME',dest='out_name',required=True,help='Output name of trained BDT')

    parser.add_argument('--seed', dest='seed',type=int,default=2,help='Numpy random seed.')
    parser.add_argument('--max_evt', dest='max_evt',type=int,default=1500000, help='Max Events to load')
    parser.add_argument('--eta', dest='eta',type=float,default=0.023, help='Learning Rate')
    parser.add_argument('--tree_number', dest='tree_number',type=int,default=1000,help='Tree Number')
    parser.add_argument('--depth', dest='depth',type=int,default=10,help='Max Tree Depth')

    arg = parser.parse_args()

    # Seed numpy's randomness
    numpy.random.seed(arg.seed)

    # Print run info
    print( f'Random seed is = {arg.seed}' )
    print( f'You set max_evt = {arg.max_evt}' )
    print( f'You set tree number = {arg.tree_number}' )
    print( f'You set max tree depth = {arg.depth}' )
    print( f'You set eta = {arg.eta}' )

    # Make Signal Container
    print( f'Loading sig_file = {arg.sig_file}' )
    sigContainer = SampleContainer(arg.sig_file,arg.max_evt,True)

    # Make Background Container
    print( f'Loading bkg_file = {arg.bkg_file}' )
    bkgContainer = SampleContainer(arg.bkg_file,arg.max_evt,False)

    t = Trainer(sigContainer, bkgContainer, arg.eta, arg.depth, arg.tree_number)

    t.train()

    t.save(arg.out_name)

    print('Done')
