"""
Core methods for both training and evaluating the BDT
"""

import ROOT

import numpy
import pickle
import xgboost
import matplotlib

ROOT.gSystem.Load('libFramework.so')

def translate(event) :
    """Translate input ROOT event object into the feature vector."""

    try :
        veto = event.EcalVeto_eat
    except :
        veto = event.EcalVeto_signal

    return [
        veto.getNReadoutHits(),
        veto.getSummedDet(),
        veto.getSummedTightIso(),
        veto.getMaxCellDep(),
        veto.getShowerRMS(),
        veto.getXStd(),
        veto.getYStd(),
        veto.getAvgLayerHit(),
        veto.getStdLayerHit(),
        veto.getDeepestLayerHit(),
        veto.getEcalBackEnergy(),
        veto.getNStraightTracks(),
        veto.getNLinRegTracks()
        ]

class SampleContainer:
    def __init__(self,filename,maxEvts,isSig):
        f = ROOT.TFile.Open(filename)
        t = f.Get('LDMX_Events')

        events = []
        events_left = maxEvts
        for event in t :
            events.append(translate(event))
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
        with open(f'{name}_weights.pkl','wb') as output :
            pickle.dump(self.gbm, output)

        if not plot_importance :
            return
    
        # Plot feature importances
        xgboost.plot_importance(self.gbm)
        matplotlib.pyplot.savefig(f'{name}_fimportance.png',
                dpi=500, bbox_inches='tight', pad_inches=0.5) # png parameters
        
