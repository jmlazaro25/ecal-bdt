import os
import ROOT as r
import numpy as np
import ROOTmanager as manager
import physTools
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
            #'nStraightTracks':   {'rtype': int,   'default': 0 },
            #'nLinregTracks':     {'rtype': int,   'default': 0 },
            #'firstNearPhLayer':  {'rtype': int,   'default': 0 },
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
        e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)

        # Photon Info from target
        e_targetHit = physTools.electronTargetSPHit(self.targetSPHits)
        if e_targetHit != None:
            g_targPos, g_targP = physTools.gammaTargetInfo(e_targetHit)
        else: g_targPos = g_targP = np.zeros(3)
        g_traj = physTools.layerIntercepts(g_targPos, g_targP) # infty >.<

    # Recoil electron momentum magnitude and angle with z-axis
    recoilPMag  = physTools.mag(  e_ecalP                 ) if e_present      else -1.0
    recoilTheta = physTools.angle(e_ecalP, units='radians') if recoilPMag > 0 else -1.0

    # Big data
    nReadoutHits = 0
    trackingHitList = []

    # Major ECal loop
    for hit in self.ecalRecHits:
        
        if hit.getEnergy() > 0:
            # idd = hitID(hit) # (maybe .getID() or .get_id(/)) and this abv cond in cxx v
            # See https://github.com/LDMX-Software/ldmx-sw/blob/23be9750016eab9cc16a4a933cd584363e05dfba/Event/include/Event/CalorimeterHit.h

            nReadoutHits += 1
            layer = physTools.layerofHitZ( hit.getZPos() )
            xy_pair = ( hit.getXPos(), hit.getYPos() )

            # Distance to electron trajectory
            if e_traj != None:
                xy_e_traj = ( e_traj[layer][0], e_traj[layer][1] )
                distance_e_traj = physTools.dist(xy_pair, xy_e_traj)
            else: distance_e_traj = -1.0

            # Distance to photon trajectory (ickily repetative; possibley)
            if g_traj != None:
                xy_g_traj = ( g_traj[layer][0], g_traj[layer][1] )
                distance_g_traj = physTools.dist(xy_pair, xy_g_traj)
            else: distance_g_traj = -1.0

            # Build MIP tracking hit list; (outside electron region or electron missing)
            """
            if distance_e_traj >= e_radii[layer] or distance_e_traj == -1.0:
                hitData = physTools.HitData()
                hitData.pos = np.array( [xy_pair[0],xy_pair[1],\
                                         physTools.layerofHitZ( hit.getZPos() )] )
                hitData.layer = layer
                trackingHitList.apped(hitData)
            """
    
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

        # Unused epAng and epSep ???
        epAng  = np.degrees( np.arccos( physTools.dot(e_norm,g_norm) ) )
        epSep  = physTools.dist( e_traj_start, g_traj_start )

    else:

        # Electron trajectory is missing so all hits in Ecal are okay to use
        # Pick trajectories so they won'trestrict tracking, far outside the Ecal

        e_traj_start = np.array([999 ,999 ,0   ])
        e_traj_end   = np.array([999 ,999 ,999 ])
        g_traj_start = np.array([1000,1000,0   ])
        g_traj_end   = np.array([1000,1000,1000])
        epAng        = 3.0 + 1.0 # Don't cut on these in this case
        epSep        = 10.0 + 1.0

    # Near photon step: Find the first layer of the ECal where a hit near the projected
    # photon trajectory is found
    # Currently unusued pending further study; performance has dropped between v9 and v12
    #firstNearPhLayer = 33;

    # In progress
    """
    if g_traj != None: # If no photon trajectory, leave this at the default
        for(std::vector<HitData>::iterator it = trackingHitList.begin(); it != trackingHitList.end(); ++it) {
            float ehDist = sqrt(pow((*it).pos.X() - photon_trajectory[(*it).layer].first, 2)
                                   + pow((*it).pos.Y() - photon_trajectory[(*it).layer].second, 2));

            if(ehDist < 8.7 && (*it).layer < firstNearPhLayer) {
                        firstNearPhLayer = (*it).layer;
    """

    ###################################
    # Reassign BDT input variables 
    ###################################
    self.tfMaker.branches['nReadoutHits'][0]           = nReadoutHits
    self.tfMaker.branches['recoilPT'][0]               = recoilPMag
    #self.tfMaker.branches['nStraightTracks'][0]       = nStraightTracks
    #self.tfMaker.branches['nLinregTracks'][0]         = nLingregTracks
    #self.tfMaker.branches['firstNearPhLayer'][0]      = firstNearPhLayer
    self.tfMaker.branches['epAng'][0]                  = epAng
    self.tfMaker.branches['epSep'][0]                  = epSep

    ###################################
    # Fill the tree with values for this event
    ###################################
    #print('hennnnllllooooo')
    self.tfMaker.tree.Fill()

if __name__ == "__main__":
    main()
