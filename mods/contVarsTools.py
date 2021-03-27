import math
import numpy as np
from mods import ecalHexReadoutTools, ecalIDTools

nEcalLayers = 34

###################################################
# Wrappers for ecal hex readout functions
###################################################

def isInShowerInnerRing(ID, probeID):
    return ecalHexReadoutTools.isNN(ID, probeID)

def isInShowerOuterRing(ID, probeID):
    return ecalHexReadoutTools.isNNN(ID, probeID)

def getCellCentroidXYPair(ID):
    return ecalHexReadoutTools.getCellCenterAbsolute(ID)

def getInnerRingIDs(ID):
    return ecalHexReadoutTools.getNN(ID)

def getOuterRingIDs(ID):
    return ecalHexReadoutTools.getNNN(ID)

###########################################
# Containment variables functions
###########################################

# Function to get the shower centroid and the shower RMS
def getShowerCentroidIDAndRMS(ecalRecHits, showerRMS):
    wgtCentroidCoords = np.zeros(2)
    sumEdep = 0
    returnCellID = ecalIDTools.emptyEcalID()

    # Calculate energy weighted centroid
    for hit in ecalRecHits:
        ID = ecalIDTools.hitIDToEcalID(hit.getID())
        cell_energy_pair = (ID, hit.getEnergy())
        centroidCoords = getCellCentroidXYPair(ID)
        wgtCentroidCoords[0] += centroidCoords[0]*cell_energy_pair[1]
        wgtCentroidCoords[1] += centroidCoords[1]*cell_energy_pair[1]
        sumEdep += cell_energy_pair[1]

    if sumEdep > 0.000001:
        wgtCentroidCoords[0] /= sumEdep
        wgtCentroidCoords[1] /= sumEdep

    # Find nearest cell to centroid
    maxDist = 1000000

    for hit in ecalRecHits:
        centroidCoords = getCellCentroidXYPair(ecalIDTools.hitIDToEcalID(hit.getID()))
        deltaR = math.sqrt((centroidCoords[0] - wgtCentroidCoords[0])**2 + (centroidCoords[1] - wgtCentroidCoords[1])**2)
        showerRMS += deltaR*hit.getEnergy()

        if deltaR < maxDist:
            maxDist = deltaR
            returnCellID = ecalIDTools.hitIDToEcalID(hit.getID())

    if sumEdep > 0:
        showerRMS /= sumEdep

    return ecalIDTools.EcalID(0, returnCellID.ecalIDTools.getModuleID(), returnCellID.ecalIDTools.getCellID()), showerRMS

# Function to fill a cell map with hit energies
def fillHitMap(ecalRecHits, cellMap):

    for hit in ecalRecHits:
        ID = ecalIDTools.hitIDToEcalID(hit.getID())

        # Emplace in C++ adds a key, value pair to a map only if the key is unique
        if all((not ecalIDTools.isSameEcalID(ID, probeID)) for probeID in cellMap):
            cellMap[ID] = hit.getEnergy()

# Function to fill a cell map with isolated hit energies
def fillIsolatedHitMap(ecalRecHits, globalCentroidID, cellMap, cellMapIso, doTight):

    for hit in ecalRecHits:
        isolatedHit = np.array([True, ecalIDTools.emptyEcalID()])
        ID = ecalIDTools.hitIDToEcalID(hit.getID())
        flatID = ecalIDTools.EcalID(0, ID.getModuleID(), ID.getCellID())

        if doTight:

            # Disregard hits that are on the centroid
            if ecalIDTools.isSameEcalID(flatID, globalCentroidID):
                continue

            # Skip hits that are on centroid inner ring
            if isInShowerInnerRing(globalCentroidID, flatID):
                continue

        # Skip hits that have a readout neighbor
        # Get neighboring cell IDs and try to look them up in the full cell map
        # These ideas are only cell/module (Must ignore layer)
        cellNbrIDs = getInnerRingIDs(ID)

        for k in range(0, 6):

            # Update neighbor ID to the current layer
            cellNbrIDs[k] = ecalIDTools.EcalID(ID.getLayerID(), cellNbrIDs[k].getModuleID(), cellNbrIDs[k].getCellID())

            # Look in cell hit map to see if it is there
            if any(ecalIDTools.isSameEcalID(cellNbrIDs[k], probeID) for probeID in cellMap):
                isolatedHit[0] = False
                isolatedHit[1] = cellNbrIDs[k]
                break

        if not isolatedHit[0]:
            continue

        # Insert isolated hit
        # Emplace in C++ adds a key, value pair to a map only if the key is unique
        if all((not ecalIDTools.isSameEcalID(ID, probeID)) for probeID in cellMapIso):
            cellMapIso[ID] = hit.getEnergy()

# Function to get the projected trajectory of a particle given a momentum and a starting position
def getTrajectory(momentum, position):
    positions = []

    for iLayer in range(0, nEcalLayers):
        posX = position[0] + (momentum[0]/momentum[2])*(ecalHexReadoutTools.getZPosition(iLayer) - position[2])
        posY = position[1] + (momentum[1]/momentum[2])*(ecalHexReadoutTools.getZPosition(iLayer) - position[2])

        positions.append((posX, posY))

    return np.array(positions)
