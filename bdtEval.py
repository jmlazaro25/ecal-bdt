import os
import sys
import numpy as np
import pickle as pkl
import xgboost as xgb
import mods.ROOTmanager as manager


def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']

    cwd = os.getcwd()
    pkl_file = cwd+'/bdt_test_0/bdt_test_0_weights.pkl'

    # TreeModel to build here
    # Note: except for the discValue, defaults don't really matter here
    branches_info = {
        'nReadoutHits':         {'rtype': int, 'default': 0.},
        'nStraightTracks':      {'rtype': int, 'default': 0.},

        'discValue_EcalVeto':   {'rtype': float, 'default': 0.5}
        }

    # Make a process
    proc = manager.TreeProcess(event_process, inlist[0], tree_name='EcalVeto', ID='onlyProc')
    proc.model = pkl.load(open(pkl_file,'rb'))
    
    # Make an output file and new tree (copied from input + discValue)
    proc.tfMaker = manager.TreeMaker(outlist[0], "EcalVeto", branches_info)
   
    # RUN
    proc.extraf = proc.tfMaker.wq # Gets executed at the end of run()
    proc.run()

    print('\nDone!\n')

def event_process(self):

    # Feature list from input tree
    # Exp: feats = [ feat_value for feat_value in self.tree ]
    # Exp: feats = [ feat_value for feat_value in self.tree ]
    feats = [
          self.tree.nReadoutHits,
          self.tree.nStraightTracks
          ]

    # Copy input tree feats to new tree
    for feat_name, feat_value in zip(self.tfMaker.branches_info, feats):
        self.tfMaker.branches[feat_name][0] = feat_value

    # Add prediction to new tree
    evtarray = np.array([feats])
    pred = float(self.model.predict(xgb.DMatrix(evtarray))[0])
    self.tfMaker.branches['discValue_EcalVeto'][0] = pred

    # Fill new tree with current event values
    self.tfMaker.tree.Fill()

if __name__ == "__main__":
    main()
