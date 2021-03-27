import ecalHexReadoutTools
import ecalIDTools

###################################################
# Wrappers for ecal hex readout functions
###################################################

def isInShowerInnerRing(ID, probeID):
    return ecalHexReadoutTools.isNN(ID, probeID)
