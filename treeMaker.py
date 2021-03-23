import os
import ROOT as r
import numpy as np
import ROOTmanager as manager
r.gSystem.Load('/home/jmlazaro/research/ldmx-sw/install/lib/libEvent.so')


def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']
    maxEvent = pdict['maxEvent']

    # TreeModel to build here
    branches_info = {
            'nReadoutHits':      {'rtype': float, 'default': 0.}
            }

    # Construct tree processes
    procs = []
    for gl, group in zip(group_labels, inlist):
        procs.append(manager.TreeProcess(event_process, group, ID=gl))

    # Process jobs
    for proc in procs:

        # Branches needed
        proc.ecalHits = proc.addBranch('EcalHit', 'EcalRecHits_v12')

        # Tree/Files(s) to make
        print('Running %s'%(proc.ID))
        proc.tfMaker = manager.TreeMaker(group_labels[procs.index(proc)]+'.root',\
                                         "EcalVeto",\
                                         branches_info,\
                                         outlist[procs.index(proc)]
                                         )

        # RUN
        proc.extraf = proc.tfMaker.wq # gets executed at the end of run()
        proc.run(maxEvents=maxEvent)

    print('\nDone!\n')


# Process an event
def event_process(self):

    # For smaller unbiased samples (maybe build into ROOTmanager)
    #if not self.event_count%4 == 0: return 0
    
    ###################################
    # Compute BDT input variables
    ###################################

    # Required reference points
    #maxTime = 50
    #nHits_back = 0
    #maxPE_1_back = 0
    #maxPE_2_back = 0
    #maxPE_3_back = 0
    #n = [0,0,0]

    # Initialize BDT input variables w/ defaults
    self.tfMaker.resetBranches()

    # Analysis

    nReadoutHits = 0
    for hit in self.ecalHits: nReadoutHits += 1

    ###################################
    # Reassign BDT input variables 
    ###################################
    self.tfMaker.branches['nReadoutHits'][0]           = nReadoutHits

    ###################################
    # Fill the tree with values for this event
    ###################################
    #print('hennnnllllooooo')
    self.tfMaker.tree.Fill()

if __name__ == "__main__":
    main()
