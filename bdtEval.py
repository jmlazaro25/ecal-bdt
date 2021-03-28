#!/usr/bin/python
import argparse
import os
import sys
import ROOT as r
import xgboost as xgb
import pickle as pkl
import numpy as np
import ROOTmanager as manager
from collections import OrderedDict


# TreeModel to build here
branches_info = {
        'nHits':      {'rtype': float, 'default': 0.},
        'centE':      {'rtype': float, 'default': 0.},
        'centPE':     {'rtype': float, 'default': 0.},
        'std_e_z':    {'rtype': float, 'default': 0.},
        'totE':       {'rtype': float, 'default': 0.},
        'totPE':      {'rtype': float, 'default': 0.},
        'maxE':       {'rtype': float, 'default': 0.},
        'maxPE':      {'rtype': float, 'default': 0.},
        'dz_e':       {'rtype': float, 'default': 0.},

        'std_e_1_e':  {'rtype': float, 'default': 0.},
        'std_e_1_r':  {'rtype': float, 'default': 0.},
        'std_e_1_pe': {'rtype': float, 'default': 0.},
        'avg_e_1_e':  {'rtype': float, 'default': 0.},
        'avg_e_1_pe': {'rtype': float, 'default': 0.},
        'tot_e_1_e':  {'rtype': float, 'default': 0.},
        'tot_e_1_pe': {'rtype': float, 'default': 0.},
        'max_e_1_e':  {'rtype': float, 'default': 0.},
        'max_e_1_pe': {'rtype': float, 'default': 0.},

        'std_e_2_e':  {'rtype': float, 'default': 0.},
        'std_e_2_r':  {'rtype': float, 'default': 0.},
        'std_e_2_pe': {'rtype': float, 'default': 0.},
        'avg_e_2_e':  {'rtype': float, 'default': 0.},
        'avg_e_2_pe': {'rtype': float, 'default': 0.},
        'tot_e_2_e':  {'rtype': float, 'default': 0.},
        'tot_e_2_pe': {'rtype': float, 'default': 0.},
        'max_e_2_e':  {'rtype': float, 'default': 0.},
        'max_e_2_pe': {'rtype': float, 'default': 0.},

        'std_e_3_e':  {'rtype': float, 'default': 0.},
        'std_e_3_r':  {'rtype': float, 'default': 0.},
        'std_e_3_pe': {'rtype': float, 'default': 0.},
        'avg_e_3_e':  {'rtype': float, 'default': 0.},
        'avg_e_3_pe': {'rtype': float, 'default': 0.},
        'tot_e_3_e':  {'rtype': float, 'default': 0.},
        'tot_e_3_pe': {'rtype': float, 'default': 0.},
        'max_e_3_e':  {'rtype': float, 'default': 0.},
        'max_e_3_pe': {'rtype': float, 'default': 0.},

        'discValue_HcalVeto': {'rtype': float, 'default': 0.5}
        }

def event_process(self):
   
    # Reset
    self.tfMaker.resetBranches()

    # Collect info for prediction
    evtarray = np.array([[
          self.tree.nHits,
          self.tree.centE,
          self.tree.centPE,
          self.tree.std_e_z,
          self.tree.totE,
          self.tree.totPE,
          self.tree.maxE,
          self.tree.maxPE,
          self.tree.dz_e,
                               
          self.tree.std_e_1_e,
          self.tree.std_e_1_r,
          self.tree.std_e_1_pe,
          self.tree.avg_e_1_e,
          self.tree.avg_e_1_pe,
          self.tree.tot_e_1_e,
          self.tree.tot_e_1_pe,
          self.tree.max_e_1_e,
          self.tree.max_e_1_pe,
                               
          self.tree.std_e_2_e,
          self.tree.std_e_2_r,
          self.tree.std_e_2_pe,
          self.tree.avg_e_2_e,
          self.tree.avg_e_2_pe,
          self.tree.tot_e_2_e,
          self.tree.tot_e_2_pe,
          self.tree.max_e_2_e,
          self.tree.max_e_2_pe,
                               
          self.tree.std_e_3_e,
          self.tree.std_e_3_r,
          self.tree.std_e_3_pe,
          self.tree.avg_e_3_e,
          self.tree.avg_e_3_pe,
          self.tree.tot_e_3_e,
          self.tree.tot_e_3_pe,
          self.tree.max_e_3_e,
          self.tree.max_e_3_pe
          ]])

    # Copy inputs
    self.tfMaker.branches['nHits'][0]           = self.tree.nHits
    self.tfMaker.branches['centE'][0]           = self.tree.centE
    self.tfMaker.branches['centPE'][0]          = self.tree.centPE
    self.tfMaker.branches['std_e_z'][0]         = self.tree.std_e_z
    self.tfMaker.branches['totE'][0]            = self.tree.totE
    self.tfMaker.branches['totPE'][0]           = self.tree.totPE
    self.tfMaker.branches['maxE'][0]            = self.tree.maxE
    self.tfMaker.branches['maxPE'][0]           = self.tree.maxPE
    self.tfMaker.branches['dz_e'][0]            = self.tree.dz_e

    self.tfMaker.branches['std_e_1_e'][0]       = self.tree.std_e_1_e
    self.tfMaker.branches['std_e_1_r'][0]       = self.tree.std_e_1_r
    self.tfMaker.branches['std_e_1_pe'][0]      = self.tree.std_e_1_pe
    self.tfMaker.branches['avg_e_1_e'][0]       = self.tree.avg_e_1_e
    self.tfMaker.branches['avg_e_1_pe'][0]      = self.tree.avg_e_1_pe
    self.tfMaker.branches['tot_e_1_e'][0]       = self.tree.tot_e_1_e
    self.tfMaker.branches['tot_e_1_pe'][0]      = self.tree.tot_e_1_pe
    self.tfMaker.branches['max_e_1_e'][0]       = self.tree.max_e_1_e
    self.tfMaker.branches['max_e_1_pe'][0]      = self.tree.max_e_1_pe

    self.tfMaker.branches['std_e_2_e'][0]       = self.tree.std_e_2_e
    self.tfMaker.branches['std_e_2_r'][0]       = self.tree.std_e_2_r
    self.tfMaker.branches['std_e_2_pe'][0]      = self.tree.std_e_2_pe
    self.tfMaker.branches['avg_e_2_e'][0]       = self.tree.avg_e_2_e
    self.tfMaker.branches['avg_e_2_pe'][0]      = self.tree.avg_e_2_pe
    self.tfMaker.branches['tot_e_2_e'][0]       = self.tree.tot_e_2_e
    self.tfMaker.branches['tot_e_2_pe'][0]      = self.tree.tot_e_2_pe
    self.tfMaker.branches['max_e_2_e'][0]       = self.tree.max_e_2_e
    self.tfMaker.branches['max_e_2_pe'][0]      = self.tree.max_e_2_pe

    self.tfMaker.branches['std_e_3_e'][0]       = self.tree.std_e_3_e
    self.tfMaker.branches['std_e_3_r'][0]       = self.tree.std_e_3_r
    self.tfMaker.branches['std_e_3_pe'][0]      = self.tree.std_e_3_pe
    self.tfMaker.branches['avg_e_3_e'][0]       = self.tree.avg_e_3_e
    self.tfMaker.branches['avg_e_3_pe'][0]      = self.tree.avg_e_3_pe
    self.tfMaker.branches['tot_e_3_e'][0]       = self.tree.tot_e_3_e
    self.tfMaker.branches['tot_e_3_pe'][0]      = self.tree.tot_e_3_pe
    self.tfMaker.branches['max_e_3_e'][0]       = self.tree.max_e_3_e
    self.tfMaker.branches['max_e_3_pe'][0]      = self.tree.max_e_3_pe

    #evtarray = np.array([evt]) Add prediction
    pred = float(self.model.predict(xgb.DMatrix(evtarray))[0])
    self.tfMaker.branches['discValue_HcalVeto'][0] = pred

    # Fill new tree with current event values
    self.tfMaker.tree.Fill()

def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']

    cwd = os.getcwd()
    pkl_file = cwd+'/bdt_0/bdt_0_weights.pkl'

    # Make a process
    proc = manager.TreeProcess(event_process, inlist[0], tree_name='HcalVeto', ID='onlyProc')
    proc.model = pkl.load(open(pkl_file,'rb'))
    
    # Make an output file and new tree (copied from input + discValue)
    proc.tfMaker = manager.TreeMaker(outlist[0], "HcalVeto", branches_info)
   
    # RUN
    proc.extraf = proc.tfMaker.wq # gets executed at the end of run()
    proc.run()

    print('\nDone!\n')

if __name__ == "__main__":
    main()
