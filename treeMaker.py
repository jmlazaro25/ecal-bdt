import os
import math
import ROOT as r
import numpy as np
from mods import ROOTmanager as manager
from mods import physTools, mipTracking
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
            # Base Vars
            'nReadoutHits':      {'rtype': int,   'default': 0 },
            'recoilPT':          {'rtype': float, 'default': 0.},
            # Segmentation Vars
            # ----------------
            # MIP tracking variables
            'nStraightTracks':   {'rtype': int,   'default': 0 },
            #'nLinregTracks':     {'rtype': int,   'default': 0 },
            'firstNearPhLayer':  {'rtype': int,   'default': 33 },
            'epAng':             {'rtype': float, 'default': 0.},
            'epSep':             {'rtype': float, 'default': 0.}
            }

    # Construct tree processes
    procs = []
    for gl, group in zip(group_labels, inlist):
        procs.append(manager.TreeProcess(event_process, group, ID=gl))

    # Process jobs
    for proc in procs:

        # Branches needed
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        proc.ecalSPHits   = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        proc.ecalRecHits  = proc.addBranch('EcalHit',       'EcalRecHits_v12')

        # Tree/Files(s) to make
        print('Running %s'%(proc.ID))
        proc.tfMaker = manager.TreeMaker(group_labels[procs.index(proc)]+'.root',\
                                         "EcalVeto",\
                                         branches_info,\
                                         outlist[procs.index(proc)]
                                         )

        # RUN
        proc.extraf = proc.tfMaker.wq # Gets executed at the end of run()
        proc.run(maxEvents=maxEvent)

    print('\nDone!\n')


# Process an event
def event_process(self):

    # For smaller unbiased samples (maybe build into ROOTmanager)
    #if not self.event_count%4 == 0: return 0

    # Initialize BDT input variables w/ defaults
    feats = {}
    for branch_name in self.tfMaker.branches_info:
        feats[branch_name] = self.tfMaker.branches_info[branch_name]['default']
    
    ###################################
    # Compute BDT input variables
    ###################################

    # Get e position and momentum, and make note of presence
    e_ecalHit = physTools.electronEcalSPHit(self.ecalSPHits)
    if e_ecalHit != None:
        e_present = True
        e_ecalPos, e_ecalP = e_ecalHit.getPosition(), e_ecalHit.getMomentum()
    else:
        e_present = False

    # Get electron and photon trajectories
    e_traj = g_traj = None
    if e_present:

        # Photon Info from target
        e_targetHit = physTools.electronTargetSPHit(self.targetSPHits)
        if e_targetHit != None:
            g_targPos, g_targP = physTools.gammaTargetInfo(e_targetHit)
        else:  # Should about never happen -> division by 0 in g_traj
            print('no e at targ!')
            g_targPos = g_targP = np.zeros(3)
        
        e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)
        g_traj = physTools.layerIntercepts(g_targPos, g_targP)

    # Recoil electron momentum magnitude and angle with z-axis
    feats['recoilPT'] = recoilPMag = physTools.mag(  e_ecalP ) if e_present      else -1.0
    recoilTheta =   physTools.angle(e_ecalP, units='radians') if recoilPMag > 0 else -1.0

    # Set electron RoC binnings
    e_radii = physTools.radius68_thetalt10_plt500
    if recoilTheta < 10 and recoilPMag >= 500: e_radii = physTools.radius68_thetalt10_pgt500
    elif recoilTheta >= 10 and recoilTheta < 20: e_radii = physTools.radius68_theta10to20
    elif recoilTheta >= 20: e_radii = physTools.radius68_thetagt20

    # Always use default binning for photon RoC
    g_radii = physTools.radius68_thetalt10_plt500

    # Big data
    trackingHitList = []

    # Major ECal loop
    for hit in self.ecalRecHits:
        
        if hit.getEnergy() > 0:
            # idd = hitID(hit) # (maybe .getID() or .get_id(/)) and this abv cond in cxx v
            # See https://github.com/LDMX-Software/ldmx-sw/blob/23be9750016eab9cc16a4a933cd584363e05dfba/Event/include/Event/CalorimeterHit.h

            feats['nReadoutHits'] += 1
            layer = physTools.layerofHitZ( hit.getZPos(), index=0 )
            xy_pair = ( hit.getXPos(), hit.getYPos() )

            # Distance to electron trajectory
            if e_traj != None:
                xy_e_traj = ( e_traj[layer][0], e_traj[layer][1] )
                distance_e_traj = physTools.dist(xy_pair, xy_e_traj)
            else: distance_e_traj = -1.0

            # Distance to photon trajectory
            if g_traj != None:
                xy_g_traj = ( g_traj[layer][0], g_traj[layer][1] )
                distance_g_traj = physTools.dist(xy_pair, xy_g_traj)
            else: distance_g_traj = -1.0

            # Build MIP tracking hit list; (outside electron region or electron missing)
            if distance_e_traj >= e_radii[layer] or distance_e_traj == -1.0:
                hitData = physTools.HitData()
                hitData.pos = np.array( [xy_pair[0],xy_pair[1],hit.getZPos() ] )
                hitData.layer = layer
                trackingHitList.append(hitData) 
    
    # MIP tracking starts here

    # Goal: Calculate 
    # nStraightTracks (Self-explanatory) 
    # nLinregTracks (Tracks found by linreg algorithm)

    # Find epAng and epSep, and prepare EP trajectory vectors
    if e_traj != None and g_traj != None:

        # Create arrays marking start and end of each trajectory
        e_traj_start = np.array([e_traj[0][0], e_traj[0][1], physTools.ecal_layerZs[0]    ])
        e_traj_end   = np.array([e_traj[-1][0], e_traj[-1][1], physTools.ecal_layerZs[-1] ])
        g_traj_start = np.array([g_traj[0][0], g_traj[0][1], physTools.ecal_layerZs[0]    ])
        g_traj_end   = np.array([g_traj[-1][0], g_traj[-1][1], physTools.ecal_layerZs[-1] ])

        evec   = e_traj_end - e_traj_start
        gvec   = g_traj_end - g_traj_start
        e_norm = physTools.unit(evec)
        g_norm = physTools.unit(gvec)

        # Unused epAng and epSep ??? And why Ang instead of dot ???
        feats['epAng'] = epAng = math.acos( physTools.dot(e_norm,g_norm) )*180.0/math.pi
        feats['epSep'] = epSep = physTools.dist( e_traj_start, g_traj_start )

    else:

        # Electron trajectory is missing so all hits in Ecal are okay to use
        # Pick trajectories so they won'trestrict tracking, far outside the Ecal

        e_traj_start   = np.array([999 ,999 ,0   ])
        e_traj_end     = np.array([999 ,999 ,999 ])
        g_traj_start   = np.array([1000,1000,0   ])
        g_traj_end     = np.array([1000,1000,1000])
        feats['epAng'] = epAng = 3.0 + 1.0 # Don't cut on these in this case
        feats['epSep'] = epSep = 10.0 + 1.0

    # Near photon step: Find the first layer of the ECal where a hit near the projected
    # photon trajectory is found
    # Currently unusued pending further study; performance has dropped between v9 and v12

    if g_traj != None: # If no photon trajectory, leave this at the default
        for hit in trackingHitList:
            if hit.layer < feats['firstNearPhLayer'] and\
                    physTools.dist( hit.pos[:2],g_traj[hit.layer]) < physTools.cellWidth:
                feats['firstNearPhLayer'] = hit.layer

    # Order hits by zpos for efficiency
    trackingHitList.sort(key=lambda hd: hd.layer, reverse=True)

    # Find MIP tracks
    feats['nStraightTracks'] = mipTracking.nStraightTracks(trackingHitList)

    # Fill the tree with values for this event
    self.tfMaker.fillEvent(feats)

if __name__ == "__main__":
    main()
