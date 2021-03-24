import physTools
import numpy as np
import os
import ROOT as r
import ROOTmanager as manager

r.gSystem.Load('/nfs/slac/g/ldmx/users/aechavez/ldmx-sw-v2.3.0-w-container/ldmx-sw/install/lib/libEvent.so')

def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']
    maxEvent = pdict['maxEvent']

    # TreeModel to build here
    branches_info = {
            'nReadoutHits': {'rtype': int, 'default': 0},
            'summedDet': {'rtype': float, 'default': 0.0},
            #'summedTightIso': {'rtype': float, 'default': 0.0},
            #'maxCellDep': {'rtype': float, 'default': 0.0},
            #'showerRMS': {'rtype': float, 'default': 0.0},
            #'xStd': {'rtype': float, 'default': 0.0},
            #'yStd': {'rtype': float, 'default': 0.0},
            #'avgLayerHit': {'rtype': float, 'default': 0.0},
            #'stdLayerHit': {'rtype': float, 'default': 0.0},
            #'deepestLayerHit': {'rtype': int, 'default': 0},
            #'ecalBackEnergy': {'rtype': float, 'default': 0.0},
            'recoilPT': {'rtype': float, 'default': 0.0}
            }

    # Construct tree processes
    procs = []
    for gl, group in zip(group_labels, inlist):
        procs.append(manager.TreeProcess(event_process, group, ID = gl))

    # Process jobs
    for proc in procs:

        # Branches needed
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        proc.ecalSPHits = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        proc.ecalHits = proc.addBranch('EcalHit', 'EcalRecHits_v12')

        # Tree/Files(s) to make
        print('Running %s'%(proc.ID))
        proc.tfMaker = manager.TreeMaker(group_labels[procs.index(proc)]+'.root',\
                                         "EcalVeto",\
                                         branches_info,\
                                         outlist[procs.index(proc)]
                                         )

        # RUN
        proc.extraf = proc.tfMaker.wq # Gets executed at the end of run()
        proc.run(maxEvents = maxEvent)

    print('\nDone!\n')


# Process an event
def event_process(self):

    # For smaller unbiased samples (maybe build into ROOTmanager)
    #if not self.event_count%4 == 0: return 0

    # Initialize BDT input variables w/ defaults
    self.tfMaker.resetBranches()

    ###################################
    # Miscellaneous constants
    ###################################

    # Number of ecal layers
    nEcalLayers = 34

    ###################################
    # Compute BDT input variables
    ###################################

    eSPHit, pSPHit = physTools.electronEcalSPHit(self.ecalSPHits), physTools.gammaEcalSPHit(self.ecalSPHits)

    if eSPHit != None:
        e_present = True
        e_pos, e_mom = eSPHit.getPosition(), eSPHit.getMomentum()
    else:
        e_present = False

    if e_present:
        recoilPT = np.sqrt(np.sum([p**2 for p in e_mom]))
    else:

        # This is not what recoilPT should default to (Figure it out later)
        recoilPT = 0.0

    ecalLayerEdepReadout = np.zeros(nEcalLayers)
    nReadoutHits = 0
    summedDet = 0.0

    for hit in self.ecalHits:
        hitID = hit.getID()
        if hit.getEnergy() > 0:
            nReadoutHits += 1
            ecalLayerEdepReadout[physTools.getLayer(hitID)] += hit.getEnergy()

    summedDet = np.sum(ecalLayerEdepReadout)

    ###################################
    # Reassign BDT input variables 
    ###################################
    self.tfMaker.branches['nReadoutHits'][0] = nReadoutHits
    self.tfMaker.branches['summedDet'][0] = summedDet
    #self.tfMaker.branches['summedTightIso'][0] = summedTightIso
    #self.tfMaker.branches['maxCellDep'][0] = maxCellDep
    #self.tfMaker.branches['showerRMS'][0] = showerRMS
    #self.tfMaker.branches['xStd'][0] = xStd
    #self.tfMaker.branches['yStd'][0] = yStd
    #self.tfMaker.branches['avgLayerHit'][0] = avgLayerHit
    #self.tfMaker.branches['stdLayerHit'][0] = stdLayerHit
    #self.tfMaker.branches['deepestLayerHit'][0] = deepestLayerHit
    #self.tfMaker.branches['ecalBackEnergy'][0] = ecalBackEnergy
    self.tfMaker.branches['recoilPT'][0] = recoilPT

    ###################################
    # Fill the tree with values for this event
    ###################################
    self.tfMaker.tree.Fill()

if __name__ == "__main__":
    main()
