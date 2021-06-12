import os
import math
import ROOT as r
import numpy as np
from mods import ROOTmanager as manager
from mods import physTools, mipTracking
cellMap = np.loadtxt('mods/cellmodule.txt')
r.gSystem.Load('libFramework.so')

# Tree model to build here
branches_info = {
        # Base variables
        'nReadoutHits'     : {'rtype': int,   'default': 0 },
        'summedDet'        : {'rtype': float, 'default': 0.},
        'summedTightIso'   : {'rtype': float, 'default': 0.},
        'maxCellDep'       : {'rtype': float, 'default': 0.},
        'showerRMS'        : {'rtype': float, 'default': 0.},
        'xStd'             : {'rtype': float, 'default': 0.},
        'yStd'             : {'rtype': float, 'default': 0.},
        'avgLayerHit'      : {'rtype': float, 'default': 0.},
        'stdLayerHit'      : {'rtype': float, 'default': 0.},
        'deepestLayerHit'  : {'rtype': int,   'default': 0 },
        'ecalBackEnergy'   : {'rtype': float, 'default': 0.},
        # Rec hit information
        'totalRecAmplitude': {'rtype': float, 'default': 0.},
        'totalRecEnergy'   : {'rtype': float, 'default': 0.},
        'nRecHits'         : {'rtype': int,   'default': 0 },
        # Sim hit information
        'totalSimEDep'     : {'rtype': float, 'default': 0.},
        'nSimHits'         : {'rtype': int,   'default': 0 },
        # Sim particle information
        'nElectrons'       : {'rtype': int,   'default': 0 },
        'nPhotons'         : {'rtype': int,   'default': 0 },
        # Noise information
        'totalNoiseEnergy' : {'rtype': float, 'default': 0.},
        'nNoiseHits'       : {'rtype': int,   'default': 0 }
}

for i in range(1, physTools.nRegions + 1):

    # Electron RoC variables
    branches_info['electronContainmentEnergy_x{}'.format(i)] = {'rtype': float, 'default': 0.}

    # Photon RoC variables
    branches_info['photonContainmentEnergy_x{}'.format(i)]   = {'rtype': float, 'default': 0.}

    # Outside RoC variables
    branches_info['outsideContainmentEnergy_x{}'.format(i)]  = {'rtype': float, 'default': 0.}
    branches_info['outsideContainmentNHits_x{}'.format(i)]   = {'rtype': int,   'default': 0 }
    branches_info['outsideContainmentXStd_x{}'.format(i)]    = {'rtype': float, 'default': 0.}
    branches_info['outsideContainmentYStd_x{}'.format(i)]    = {'rtype': float, 'default': 0.}

# Branches needed by TTrees storing hit-by-hit/particle-by-particle information
recVsSimHitBranches = {
    'recHitAmplitude': {'address': np.zeros(1, dtype = float), 'rtype': float},
    'recHitEnergy'   : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'simHitMatchEDep': {'address': np.zeros(1, dtype = float), 'rtype': float},
}

recHitBranches = {
    'recHitX'        : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'recHitY'        : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'recHitZ'        : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'recHitLayer'    : {'address': np.zeros(1, dtype = int  ), 'rtype': int  },
    'recHitAmplitude': {'address': np.zeros(1, dtype = float), 'rtype': float},
    'recHitEnergy'   : {'address': np.zeros(1, dtype = float), 'rtype': float},
}

simHitBranches = {
    'simHitX'     : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'simHitY'     : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'simHitZ'     : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'simHitLayer' : {'address': np.zeros(1, dtype = int  ), 'rtype': int  },
    'simHitEDep'  : {'address': np.zeros(1, dtype = float), 'rtype': float},
}

simParticleBranches = {
    'vertX'       : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'vertY'       : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'vertZ'       : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'energy'      : {'address': np.zeros(1, dtype = float), 'rtype': float},
    'pdgID'       : {'address': np.zeros(1, dtype = int  ), 'rtype': int  },
    'nParents'    : {'address': np.zeros(1, dtype = int  ), 'rtype': int  },
    'nDaughters'  : {'address': np.zeros(1, dtype = int  ), 'rtype': int  },
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
        proc.simParticles = proc.addBranch('SimParticle', 'SimParticles_v12')

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
                                        "EcalInfo",\
                                        branches_info,\
                                        outlist[procs.index(proc)]
                                        )

        # TTree for rec/sim hit comparison
        proc.recVsSimHitInfo = r.TTree('RecVsSimHitInfo', 'Information for rec/sim hit comparison')

        for branch in recVsSimHitBranches:

            if str(recVsSimHitBranches[branch]['rtype']) == "<class 'float'>"\
              or str(recVsSimHitBranches[branch]['rtype']) == "<type 'float'>":
                proc.recVsSimHitInfo.Branch(branch, recVsSimHitBranches[branch]['address'], branch + '/D')

            elif str(recVsSimHitBranches[branch]['rtype']) == "<class 'int'>"\
              or str(recVsSimHitBranches[branch]['rtype']) == "<type 'int'>":
                proc.recVsSimHitInfo.Branch(branch, recVsSimHitBranches[branch]['address'], branch + '/I')

        # TTree for rec hit information
        proc.recHitInfo = r.TTree('RecHitInfo', 'Rec hit information')

        for branch in recHitBranches:

            if str(recHitBranches[branch]['rtype']) == "<class 'float'>"\
              or str(recHitBranches[branch]['rtype']) == "<type 'float'>":
                proc.recHitInfo.Branch(branch, recHitBranches[branch]['address'], branch + '/D')

            elif str(recHitBranches[branch]['rtype']) == "<class 'int'>"\
              or str(recHitBranches[branch]['rtype']) == "<type 'int'>":
                proc.recHitInfo.Branch(branch, recHitBranches[branch]['address'], branch + '/I')

        # TTree for sim hit information
        proc.simHitInfo = r.TTree('SimHitInfo', 'Sim hit information')

        for branch in simHitBranches:

            if str(simHitBranches[branch]['rtype']) == "<class 'float'>"\
              or str(simHitBranches[branch]['rtype']) == "<type 'float'>":
                proc.simHitInfo.Branch(branch, simHitBranches[branch]['address'], branch + '/D')

            elif str(simHitBranches[branch]['rtype']) == "<class 'int'>"\
              or str(simHitBranches[branch]['rtype']) == "<type 'int'>":
                proc.simHitInfo.Branch(branch, simHitBranches[branch]['address'], branch + '/I')

        # TTree for sim particle information
        proc.simParticleInfo = r.TTree('SimParticleInfo', 'Sim particle information')

        for branch in simParticleBranches:

            if str(simParticleBranches[branch]['rtype']) == "<class 'float'>"\
              or str(simParticleBranches[branch]['rtype']) == "<type 'float'>":
                proc.simParticleInfo.Branch(branch, simParticleBranches[branch]['address'], branch + '/D')

            elif str(simParticleBranches[branch]['rtype']) == "<class 'int'>"\
              or str(simParticleBranches[branch]['rtype']) == "<type 'int'>":
                proc.simParticleInfo.Branch(branch, simParticleBranches[branch]['address'], branch + '/I')

        # Gets executed at the end of run()
        proc.extrafs = []
        for tfMaker in proc.tfMakers:
            proc.extrafs.append(proc.recVsSimHitInfo.Write)
            proc.extrafs.append(proc.recHitInfo.Write)
            proc.extrafs.append(proc.simHitInfo.Write)
            proc.extrafs.append(proc.simParticleInfo.Write)
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

    for i in range(0, physTools.nRegions):
        feats['electronContainmentEnergy_x{}'.format(i + 1)] = self.ecalVeto.getElectronContainmentEnergy()[i]
        feats['photonContainmentEnergy_x{}'.format(i + 1)  ] = self.ecalVeto.getPhotonContainmentEnergy()[i]
        feats['outsideContainmentEnergy_x{}'.format(i + 1) ] = self.ecalVeto.getOutsideContainmentEnergy()[i]
        feats['outsideContainmentNHits_x{}'.format(i + 1)  ] = self.ecalVeto.getOutsideContainmentNHits()[i]
        feats['outsideContainmentXStd_x{}'.format(i + 1)   ] = self.ecalVeto.getOutsideContainmentXStd()[i]
        feats['outsideContainmentYStd_x{}'.format(i + 1)   ] = self.ecalVeto.getOutsideContainmentYStd()[i]
    
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
    else:

        if e_ecalHit != None:
            e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)

        if e_targetHit != None:
            g_traj = physTools.layerIntercepts(g_targPos, g_targP)

    #################################################
    # Quantities needed for sample analysis
    #################################################

    # Global rec hit information
    feats['totalRecAmplitude'] = sum([recHit.getAmplitude() for recHit in self.ecalRecHits])
    feats['totalRecEnergy'] = sum([recHit.getEnergy() for recHit in self.ecalRecHits])
    feats['nRecHits'] = len([recHit for recHit in self.ecalRecHits])

    # Global sim hit information
    feats['totalSimEDep'] = sum([simHit.getEdep() for simHit in self.ecalSimHits])
    feats['nSimHits'] = len([simHit for simHit in self.ecalSimHits])

    # Global sim particle information
    feats['nElectrons'] = len([simParticle for trackID, simParticle in self.simParticles if (simParticle.getPdgID() == 11)])
    feats['nPhotons'] = len([simParticle for trackID, simParticle in self.simParticles if (simParticle.getPdgID() == 22)])

    # Hit-by-hit rec hit information
    for recHit in self.ecalRecHits:
        recHitBranches['recHitX'        ]['address'][0] = recHit.getXPos()
        recHitBranches['recHitY'        ]['address'][0] = recHit.getYPos()
        recHitBranches['recHitZ'        ]['address'][0] = recHit.getZPos()
        recHitBranches['recHitLayer'    ]['address'][0] = physTools.ecal_layer(recHit)
        recHitBranches['recHitAmplitude']['address'][0] = recHit.getAmplitude()
        recHitBranches['recHitEnergy'   ]['address'][0] = recHit.getEnergy()

        self.recHitInfo.Fill()

    # Hit-by-hit sim hit information
    for simHit in self.ecalSimHits:
        simHitBranches['simHitX'     ]['address'][0] = simHit.getPosition()[0]
        simHitBranches['simHitY'     ]['address'][0] = simHit.getPosition()[1]
        simHitBranches['simHitZ'     ]['address'][0] = simHit.getPosition()[2]
        simHitBranches['simHitLayer' ]['address'][0] = physTools.ecal_layer(simHit)
        simHitBranches['simHitEDep'  ]['address'][0] = simHit.getEdep()

        self.simHitInfo.Fill()

    # Particle-by-particle sim particle information
    for trackID, simParticle in self.simParticles:
        simParticleBranches['vertX'       ]['address'][0] = simParticle.getVertex()[0]
        simParticleBranches['vertY'       ]['address'][0] = simParticle.getVertex()[1]
        simParticleBranches['vertZ'       ]['address'][0] = simParticle.getVertex()[2]
        simParticleBranches['energy'      ]['address'][0] = simParticle.getEnergy()
        simParticleBranches['pdgID'       ]['address'][0] = simParticle.getPdgID()
        simParticleBranches['nParents'    ]['address'][0] = len(simParticle.getParents())
        simParticleBranches['nDaughters'  ]['address'][0] = len(simParticle.getDaughters())

        self.simParticleInfo.Fill()

    # Sort the ecal rec hits and sim hits by hitID
    ecalRecHitsSorted = [hit for hit in self.ecalRecHits]
    ecalRecHitsSorted.sort(key = lambda hit : hit.getID())
    ecalSimHitsSorted = [hit for hit in self.ecalSimHits]
    ecalSimHitsSorted.sort(key = lambda hit : hit.getID())

    # For-loop to get noise info and info for 2D distributions
    for recHit in ecalRecHitsSorted:

        # If the noise flag is set, count the hit as a noise hit
        if recHit.isNoise():
            feats['totalNoiseEnergy'] += recHit.getEnergy()
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

        # Fill the TTree
        recVsSimHitBranches['recHitAmplitude']['address'][0] = recHit.getAmplitude()
        recVsSimHitBranches['recHitEnergy'   ]['address'][0] = recHit.getEnergy()
        recVsSimHitBranches['simHitMatchEDep']['address'][0] = simHitMatchEDep

        self.recVsSimHitInfo.Fill()

        # If no matching sim hit exists, count the hit as a noise hit
        if (not recHit.isNoise()) and (nSimHitMatch == 0):
            feats['totalNoiseEnergy'] += recHit.getEnergy()
            feats['nNoiseHits'] += 1

    # Fill the tree (according to fiducial category) with values for this event
    if not self.separate:
        self.tfMakers['unsorted'].fillEvent(feats)
    else:
        if e_fid and g_fid: self.tfMakers['egin'].fillEvent(feats)
        elif e_fid and not g_fid: self.tfMakers['ein'].fillEvent(feats)
        elif not e_fid and g_fid: self.tfMakers['gin'].fillEvent(feats)
        else: self.tfMakers['none'].fillEvent(feats)

if __name__ == "__main__":
    main()
