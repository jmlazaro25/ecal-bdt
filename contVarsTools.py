import ecalHexReadoutTools
import ecalIDTools

###################################################
# Wrappers for ecal hex readout functions
###################################################

def isInShowerInnerRing(centroidID, probeID):
    return ecalHexReadoutTools.isNN(centroidID, probeID)
