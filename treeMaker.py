import os
import ROOT as r
import numpy as np
import ROOTmanager as manager
import acceptTools
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
            'nReadoutHits':      {'rtype': int,   'default': 0},
            'recoilPT':          {'rtype': float, 'default': 0.}
            }

    # Construct tree processes
    procs = []
    for gl, group in zip(group_labels, inlist):
        procs.append(manager.TreeProcess(event_process, group, ID=gl))

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
        proc.extraf = proc.tfMaker.wq # gets executed at the end of run()
        proc.run(maxEvents=maxEvent)

    print('\nDone!\n')


# Process an event
def event_process(self):

    # For smaller unbiased samples (maybe build into ROOTmanager)
    #if not self.event_count%4 == 0: return 0

    # Initialize BDT input variables w/ defaults
    self.tfMaker.resetBranches()
    
    ###################################
    # Compute BDT input variables
    ###################################

    # Get number of readout hits
    nReadoutHits = 0
    for hit in self.ecalHits:
        if hit.getEnergy() > 0:
            nReadoutHits += 1

    # Find scoring plane hits
    eSPHit, gSPHit = acceptTools.elec_gamma_ecalSPHits(self.ecalSPHits)

    # Get e/g position and momentum and make note of presence
    if eSPHit != None:
        e_present = True
        e_pos, e_mom = eSPHit.getPosition(), eSPHit.getMomentum()
    else:
        e_present = False

    """
    if gSPHit != None:
        g_present = True
        g_pos, g_mom = gSPHit.getPosition(), gSPHit.getMomentum()
        if acceptTools.angle(g_mom) > self.max_angle:
            self.too_wide += 1
            return
    else:
        g_present = False
    """

    # Calculate recoilPT
    if e_present:
        recoilPT = np.sqrt(np.sum([pi**2 for pi in e_mom]))
    else:
        recoilPT = 0

    ###################################
    # Reassign BDT input variables 
    ###################################
    self.tfMaker.branches['nReadoutHits'][0]           = nReadoutHits
    self.tfMaker.branches['recoilPT'][0]               = recoilPT

    ###################################
    # Fill the tree with values for this event
    ###################################
    #print('hennnnllllooooo')
    self.tfMaker.tree.Fill()

if __name__ == "__main__":
    main()
