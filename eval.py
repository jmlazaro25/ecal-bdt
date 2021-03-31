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

    pkl_file = os.getcwd()+'/bdt_test_1/bdt_test_1_weights.pkl'

    # TreeModel to build here
    # Note: except for the discValue, defaults don't really matter here
    branches_info = {
        'nReadoutHits':         {'rtype': int,   'default': 0 }, # 0
        'summedDet':            {'rtype': float, 'default': 0.}, # 1
        'summedTightIso':       {'rtype': float, 'default': 0.}, # 2
        'maxCellDep':           {'rtype': float, 'default': 0.}, # 3
        'showerRMS':            {'rtype': float, 'default': 0.}, # 4

        'xStd':                 {'rtype': float, 'default': 0.}, # 5
        'yStd':                 {'rtype': float, 'default': 0.}, # 6
        'avgLayerHit':          {'rtype': float, 'default': 0.}, # 7
        'stdLayerHit':          {'rtype': float, 'default': 0.}, # 8
        'deepestLayerHit':      {'rtype': int,   'default': 0 }, # 9

        'ecalBackEnergy':       {'rtype': float, 'default': 0.}, # 10
        'recoilPT':             {'rtype': float, 'default': 0.}, # 11
        'nStraightTracks':      {'rtype': int,   'default': 0 }, # 12

        'discValue_EcalVeto':   {'rtype': float, 'default': 0.5}
        }

    # Make a process
    proc = manager.TreeProcess(event_process, inlist[0], tree_name='EcalVeto', ID='onlyProc')
    proc.model = pkl.load(open(pkl_file,'rb'))
    
    # Make an output file and new tree (copied from input + discValue)
    proc.tfMaker = manager.TreeMaker(outlist[0], 'EcalVeto', branches_info)
   
    # RUN
    proc.extraf = proc.tfMaker.wq # Gets executed at the end of run()
    proc.run()

    print('\nDone!\n')

def event_process(self):

    # Feature list from input tree
    # Exp: feats = [ feat_value for feat_value in self.tree~ ]
    feats = [
            self.tree.nReadoutHits,
            self.tree.summedDet,
            self.tree.summedTightIso,
            self.tree.maxCellDep,
            self.tree.showerRMS,

            self.tree.xStd,
            self.tree.yStd,
            self.tree.avgLayerHit,
            self.tree.stdLayerHit,
            self.tree.deepestLayerHit,

            self.tree.ecalBackEnergy,
            self.tree.recoilPT,
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
