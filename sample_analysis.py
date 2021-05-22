import os
import math
import ROOT as r
import numpy as np
from mods import ROOTmanager as manager
from mods import physTools, mipTracking
cellMap = np.loadtxt('mods/cellmodule.txt')

r.gSystem.Load('/nfs/slac/g/ldmx/users/aechavez/ldmx-sw-v3.0.0-w-container/ldmx-sw/install/lib/libFramework.so')

# TreeModel to build here
branches_info = {
        # Base variables
        'nReadoutHits':    {'rtype': int,   'default': 0 },
        'summedDet':       {'rtype': float, 'default': 0.},
        'summedTightIso':  {'rtype': float, 'default': 0.},
        'maxCellDep':      {'rtype': float, 'default': 0.},
        'showerRMS':       {'rtype': float, 'default': 0.},
        'xStd':            {'rtype': float, 'default': 0.},
        'yStd':            {'rtype': float, 'default': 0.},
        'avgLayerHit':     {'rtype': float, 'default': 0.},
        'stdLayerHit':     {'rtype': float, 'default': 0.},
        'deepestLayerHit': {'rtype': int,   'default': 0 },
        'ecalBackEnergy':  {'rtype': float, 'default': 0.},
        # Hit information
        'recHitAmplitude': {'rtype': float, 'default': 0.},
        'recHitEnergy':    {'rtype': float, 'default': 0.},
        'nRecHits':        {'rtype': int,   'default': 0 },
        # Sim hit information
        'simHitEDep':      {'rtype': float, 'default': 0.},
        'nSimHits':        {'rtype': int,   'default': 0 },
        # Noise information
        'noiseEnergy':     {'rtype': float, 'default': 0.},
        'nNoiseHits':      {'rtype': int,   'default': 0 }
}

def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    batch_mode = pdict['batch']
    separate = pdict['separate']
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']
    startEvent = pdict['startEvent']
    maxEvents = pdict['maxEvents']
    # Should maybe put in parsing eventually and make event_process *arg

    # Construct tree processes
    procs = []
    for gl, group in zip(group_labels,inlist):
        procs.append( manager.TreeProcess(event_process, group,
                                          ID=gl, batch=batch_mode, pfreq=100) )

    # Process jobs
    for proc in procs:

        # Move into appropriate scratch dir
        os.chdir(proc.tmp_dir)

        # Branches needed
        proc.ecalVeto     = proc.addBranch('EcalVetoResult', 'EcalVeto_v12')
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        proc.ecalSPHits   = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        proc.ecalRecHits  = proc.addBranch('EcalHit', 'EcalRecHits_v12')
        proc.ecalSimHits  = proc.addBranch('SimCalorimeterHit', 'EcalSimHits_v12')

        # Tree/Files(s) to make
        print('\nRunning %s'%(proc.ID))

        proc.separate = separate

        proc.tfMakers = {'unsorted': None}
        if proc.separate:
            proc.tfMakers = {
                'egin': None,
                'ein': None,
                'gin': None,
                'none': None
                }

        for tfMaker in proc.tfMakers:
            proc.tfMakers[tfMaker] = manager.TreeMaker(group_labels[procs.index(proc)]+\
                                        '_{}.root'.format(tfMaker),\
                                        "SampleAnalysis",\
                                        branches_info,\
                                        outlist[procs.index(proc)]
                                        )

        # N-tuples for 2D distributions
        proc.amp_vs_edep_tup = r.TNtuple('recHitAmplitude_vs_simHitEDep', 'Rec Hit Amplitude vs. Sim Hit EDep', 'simHitEDep:recHitAmplitude')
        proc.energy_vs_edep_tup = r.TNtuple('recHitEnergy_vs_simHitEDep', 'Rec Hit Energy vs. Sim Hit EDep', 'simHitEDep:recHitEnergy')

        # Simple function used to save the n-tuples
        def saveTuples():
            proc.amp_vs_edep_tup.Write()
            proc.energy_vs_edep_tup.Write()

        # Gets executed at the end of run()
        proc.extrafs = []
        for tfMaker in proc.tfMakers:
            proc.extrafs.append(saveTuples)
        for tfMaker in proc.tfMakers:
            proc.extrafs.append(proc.tfMakers[tfMaker].wq)

        # RUN
        proc.run(strEvent=startEvent, maxEvents=maxEvents)

    # Remove scratch directory if there is one
    if not batch_mode:     # Don't want to break other batch jobs when one finishes
        manager.rmScratch()

    print('\nDone!\n')


# Process an event
def event_process(self):

    # Initialize BDT input variables w/ defaults
    feats = next(iter(self.tfMakers.values())).resetFeats()

    #########################################
    # Assign pre-computed variables
    #########################################

    feats['nReadoutHits']       = self.ecalVeto.getNReadoutHits()
    feats['summedDet']          = self.ecalVeto.getSummedDet()
    feats['summedTightIso']     = self.ecalVeto.getSummedTightIso()
    feats['maxCellDep']         = self.ecalVeto.getMaxCellDep()
    feats['showerRMS']          = self.ecalVeto.getShowerRMS()
    feats['xStd']               = self.ecalVeto.getXStd()
    feats['yStd']               = self.ecalVeto.getYStd()
    feats['avgLayerHit']        = self.ecalVeto.getAvgLayerHit()
    feats['stdLayerHit']        = self.ecalVeto.getStdLayerHit()
    feats['deepestLayerHit']    = self.ecalVeto.getDeepestLayerHit() 
    feats['ecalBackEnergy']     = self.ecalVeto.getEcalBackEnergy()
    
    ###################################
    # Determine event type
    ###################################

    # Get e position and momentum from EcalSP
    e_ecalHit = physTools.electronEcalSPHit(self.ecalSPHits)
    if e_ecalHit != None:
        e_ecalPos, e_ecalP = e_ecalHit.getPosition(), e_ecalHit.getMomentum()

    # Photon Info from targetSP
    e_targetHit = physTools.electronTargetSPHit(self.targetSPHits)
    if e_targetHit != None:
        g_targPos, g_targP = physTools.gammaTargetInfo(e_targetHit)
    else:  # Should about never happen -> division by 0 in g_traj
        # Print statement commented out for now because it's a little noisy
        # print('No e at target SP!')
        g_targPos = g_targP = np.zeros(3)

    # Get electron and photon trajectories AND
    # Fiducial categories (filtered into different output trees)
    e_traj = g_traj = None
    if self.separate:
        e_fid = g_fid = False

        if e_ecalHit != None:
            e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)
            for cell in cellMap:
                if physTools.dist( cell[1:], e_traj[0] ) <= physTools.cell_radius:
                    e_fid = True
                    break

        if e_targetHit != None:
            g_traj = physTools.layerIntercepts(g_targPos, g_targP)
            for cell in cellMap:
                if physTools.dist( cell[1:], g_traj[0] ) <= physTools.cell_radius:
                    g_fid = True
                    break

    #################################################
    # Quantities needed for sample analysis
    #################################################

    # Sort the ecal rec hits and sim hits by hitID
    ecalRecHitsSorted = [hit for hit in self.ecalRecHits]
    ecalRecHitsSorted.sort(key = lambda hit : hit.getID())
    ecalSimHitsSorted = [hit for hit in self.ecalSimHits]
    ecalSimHitsSorted.sort(key = lambda hit : hit.getID())

    # Loop over hits to get hit information
    for recHit in ecalRecHitsSorted:

        # Add to totals
        feats['recHitAmplitude'] += recHit.getAmplitude()
        feats['recHitEnergy'] += recHit.getEnergy()
        feats['nRecHits'] += 1

        # If the noise flag is set, count the hit as a noise hit
        if recHit.isNoise():
            feats['noiseEnergy'] += recHit.getEnergy()
            feats['nNoiseHits'] += 1

        # Otherwise, check for a sim hit whose hitID matches
        nSimHitMatch = 0
        simHitMatchEDep = 0

        for simHit in ecalSimHitsSorted:

            if simHit.getID() == recHit.getID():
                simHitMatchEDep += simHit.getEdep()
                nSimHitMatch += 1

            elif simHit.getID() > recHit.getID():
                break

        # Fill the n-tuples
        self.amp_vs_edep_tup.Fill(simHitMatchEDep, recHit.getAmplitude())
        self.energy_vs_edep_tup.Fill(simHitMatchEDep, recHit.getEnergy())

        # If no matching sim hit exists, count the hit as a noise hit
        if (not recHit.isNoise()) and (nSimHitMatch == 0):
            feats['noiseEnergy'] += recHit.getEnergy()
            feats['nNoiseHits'] += 1

    # Loop over sim hits to get sim hit information
    for simHit in ecalSimHitsSorted:
        feats['simHitEDep'] += simHit.getEdep()
        feats['nSimHits'] += 1

    # Fill the tree (according to fiducial category) with values for this event
    #print(e_fid, g_fid)
    if not self.separate:
        self.tfMakers['unsorted'].fillEvent(feats)
    else:
        if e_fid and g_fid: self.tfMakers['egin'].fillEvent(feats)
        elif e_fid and not g_fid: self.tfMakers['ein'].fillEvent(feats)
        elif not e_fid and g_fid: self.tfMakers['gin'].fillEvent(feats)
        else: self.tfMakers['none'].fillEvent(feats)

if __name__ == "__main__":
    main()
