import ecalIDTools
import math
import numpy as np

gap = 1.5
moduler = 85.0
layerZPositions = [7.850, 13.300, 26.400, 33.500, 47.950, 56.550, 72.250, 81.350, 97.050, 106.150,
                   121.850, 130.950, 146.650, 155.750, 171.450, 180.550, 196.250, 205.350, 221.050,
                   230.150, 245.850, 254.950, 270.650, 279.750, 298.950, 311.550, 330.750, 343.350,
                   362.550, 375.150, 394.350, 406.950, 426.150, 438.750]
ecalFrontZ = 240.5
nCellRHeight = 35.3

moduleR = moduler*(2/math.sqrt(3))
cellR = 2*moduler/nCellRHeight
cellr = (0.5*math.sqrt(3))*cellR

#########################
# Miscellaneous
#########################

# Honeycomb function needed to build the cell position map (Taken from TH2Poly::Honeycomb in ROOT)
def honeycomb(xstart, ystart, a, k, s):
    hexaVerts = []
    x = np.full(6, None)
    y = np.full(6, None)
    xloop = xstart
    yloop = ystart + 0.5*a

    for i in range(0, s):
        xtemp = xloop

        if i%2 == 0:
            numberOfHexagonsInTheRow = k
        else:
            numberOfHexagonsInTheRow = k - 1

        for j in range(0, numberOfHexagonsInTheRow):
            x[0] = xtemp
            y[0] = yloop
            x[1] = x[0]
            y[1] = y[0] + a
            x[2] = x[1] + 0.5*a*math.sqrt(3)
            y[2] = y[1] + 0.5*a
            x[3] = x[2] + 0.5*a*math.sqrt(3)
            y[3] = y[1]
            x[4] = x[3]
            y[4] = y[0]
            x[5] = x[2]
            y[5] = y[4] - 0.5*a

            hexaVerts.append(np.array([(x[k], y[k]) for k in range(0, 6)]))

            xtemp += a*math.sqrt(3)

        if i%2 == 0:
            xloop += 0.5*a*math.sqrt(3)
        else:
            xloop -= 0.5*a*math.sqrt(3)

        yloop += 1.5*a

    return np.array(hexaVerts)

# Function used to check whether a hexagon from the honeycomb falls inside the module
def isInside(normX, normY):
    normX = abs(normX)
    normY = abs(normY)
    xvec = -1
    yvec = -1/math.sqrt(3)
    xref = 0.5
    yref = 0.5*math.sqrt(3)

    if (normX > 1) or (normY > yref):
        return False

    dotProd = xvec*(normX - xref) + yvec*(normY - yref)

    return (dotProd > 0)

###############################################################
# Position maps needed for ecal hex readout functions
###############################################################

# Subroutine to build the module position map
def buildModuleMap():

    # Module IDs are 0 for ecal center, 1 at 12 o'clock, and clockwise till 6 at 11 o'clock
    modulePositionMap = {}
    modulePositionMap[0] = (0.0, 0.0)

    for i in range(1, 7):
        x = (2*moduler + gap)*math.sin((i - 1)*(math.pi/3))
        y = (2*moduler + gap)*math.cos((i - 1)*(math.pi/3))

        modulePositionMap[i] = (x, y)

    return modulePositionMap

# Subroutine to build the cell position map
def buildCellMap():

    # Strategy: Use honeycomb function to build large hexagonal grid, then copy from it the polygons which cover a
    # module. Make hexagonal grid (Boundary is rectangle) larger than the module
    cellPositionMap = {}
    gridMinX = -cellr
    gridMinY = 0.0
    numXCells = 1
    numYCells = 0

    while gridMinX > -moduleR:

        # Decrement x by cell center-to-flat diameter
        gridMinX -= 2*cellr
        numXCells += 1

    while gridMinY > -moduler:

        # Decrement y by cell center-to-corner radius
        # Alternate between a full corner-to-corner diameter and a side of a cell (Center-to-corner radius)
        if numYCells%2 == 0:
            gridMinY -= cellR
        else:
            gridMinY -= 2*cellR

        numYCells += 1

    # Only counted one half of the cells
    numXCells *= 2
    numYCells *= 2

    hexaVerts = honeycomb(gridMinX, gridMinY, cellR, numXCells, numYCells)

    # Copy cells lying within module boundaries to a module grid
    ecalMapID = 0

    # For loop over hexagons (In EcalHexReadout.cxx, this is a while loop over TH2PolyBin)
    for hexa in hexaVerts:

        # Decide whether to copy polygon to new map
        # Use all vertices in case of cut-off edge polygons
        numVerticesInside = 0
        vertex_x = np.full(6, None)
        vertex_y = np.full(6, None)
        isinside = np.full(6, None)

        for i in range(0, 6):
            vertex_x[i] = hexa[i][0]
            vertex_y[i] = hexa[i][1]
            isinside[i] = isInside(vertex_x[i]/moduleR, vertex_y[i]/moduleR)

            if isinside[i]:
                numVerticesInside += 1

        if numVerticesInside > 1:

            hexaXMax = sorted(hexa, key = lambda v: v[0])[-1][0]
            hexaXMin = sorted(hexa, key = lambda v: v[0])[0][0]
            hexaYMax = sorted(hexa, key = lambda v: v[1])[-1][1]
            hexaYMin = sorted(hexa, key = lambda v: v[1])[0][1]

            x = 0.5*(hexaXMax + hexaXMin)
            y = 0.5*(hexaYMax + hexaYMin)

            # Save cell location as center of entire hexagon
            cellPositionMap[ecalMapID] = (x, y)

            # Increment cell ID
            ecalMapID += 1

    return cellPositionMap

# Build the module position map
modulePositionMap = buildModuleMap()

# Build the cell position map
cellPositionMap = buildCellMap()

# Subroutine to build the cell module position map
def buildCellModuleMap():
    cellModulePositionMap = {}

    for moduleID in modulePositionMap:
        moduleX = modulePositionMap[moduleID][0]
        moduleY = modulePositionMap[moduleID][1]

        for cellID in cellPositionMap:
            cellX = cellPositionMap[cellID][0]
            cellY = cellPositionMap[cellID][1]
            x = cellX + moduleX
            y = cellY + moduleY
            cellModulePositionMap[ecalIDTools.EcalID(0, moduleID, cellID)] = (x, y)

    return cellModulePositionMap

# Build the cell module position map
cellModulePositionMap = buildCellModuleMap()

######################################
# Ecal hex readout functions
######################################

# Function to get the layerZ associated with a layerID
def getZPosition(layerID):
    return ecalFrontZ + layerZPositions[layerID]

# Function to get the position of a cell associated with an ecalID
def getCellCenterAbsolute(ID):
    flatID = [flatID_ for flatID_ in cellModulePositionMap if ecalIDTools.isFlatEcalID(ID, flatID_)][0]
    return cellModulePositionMap[flatID]
