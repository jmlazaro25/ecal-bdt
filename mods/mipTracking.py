import math
import numpy as np
from mods import physTools


# NOTE: Don't forget to order hits by reverse zpos before using any of this
# This is assumed to make use of some algorithm short cuts

##########################
# Straight tracks
##########################

def nStraightTracks(trackingHitList):

    nTracks = 0

    # Seed a track with each hit
    iHit = 0
    while iHit < len(trackingHitList):
        track = 34*[999]
        track[0] = iHit
        currentHit = iHit
        trackLen = 1        

        # Search for hits in next two layers
        jHit = 0
        while jHit < len(trackingHitList):

            if trackingHitList[jHit].layer == trackingHitList[currentHit].layer or\
                    trackingHitList[jHit].layer > trackingHitList[currentHit].layer + 2:
                jHit += 1 # Don't keep checking this hit over and over again
                continue # Continue if not in the right range

            # If it's also directly behind the current hit, add it to the current track
            if trackingHitList[jHit].pos[:1] == trackingHitList[currentHit].pos[:1]:
                track[trackLen] = jHit
                currentHit = jHit # Update end of track
                trackLen += 1

            jHit += 1 # Move j along

        # Confirm if track is valid
        if trackLen >= 2: # Set min track length
            
            # Make sure the track is near the photon trajectory and away from the electron
            # closest_e = physTools.distTwoLines()
            #
            # more condition lines coming
            #

            # If valid track is found, remove hits in track from hitList
            for kHit in range(trackLen):
                trackingHitList.pop( track[kHit] - kHit) 

            # nStraightTracks++
            nTracks += 1

            # Decrease iHit because the *current" seed will have been removed
            iHit -= 1

        iHit += 1 # Move iHit along

    return nTracks
