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

# Function to build the module position map
def buildModuleMap():

    # Module IDs are 0 for ecal center, 1 at 12 o'clock, and clockwise till 6 at 11 o'clock
    modulePositionMap = {}
    modulePositionMap[0] = (0.0, 0.0)

    for i in range(1, 7):
        x = (2*moduler + gap)*math.sin((i - 1)*(math.pi/3))
        y = (2*moduler + gap)*math.cos((i - 1)*(math.pi/3))

        modulePositionMap[i] = (x, y)

    return modulePositionMap

# Function to build the cell position map
def buildCellMap():

    # Strategy: Use honeycomb function to build large hexagonal grid, then copy from it the polygons which cover a module
    # Make hexagonal grid (Boundary is rectangle) larger than the module
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
    ecalPolyVerts = []

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

            # Include this cell if more than one of its vertices is inside the module hexagon
            actual_x = np.full(8, None)
            actual_y = np.full(8, None)
            num_vertices = 0

            if numVerticesInside < 6:

                # This cell is stradling the edge of the module and is NOT cleanly cut by module edge
                # Loop through vertices
                for i in range(0, 6):

                    if i == 5:
                        up = 0
                    else:
                        up = i + 1

                    if i == 0:
                        dn = 5
                    else:
                        dn = i - 1

                    if (isinside[i]) and ((not isinside[up]) or (not isinside[dn])):

                        # This vertex is inside the module hexagon and is adjacent to a vertex outside
                        # Have to project this vertex onto the nearest edge of the module hexagon
                        if vertex_x[i] < -0.5*moduleR:

                            # Sloped edge on negative-x side
                            edge_origin_x = -moduleR
                            edge_origin_y = 0.0
                            edge_dest_x = -0.5*moduleR
                            edge_dest_y = moduler

                        elif vertex_x[i] > 0.5*moduleR:

                            # Sloped edge on positive-x side
                            edge_origin_x = 0.5*moduleR
                            edge_origin_y = moduler
                            edge_dest_x = moduleR
                            edge_dest_y = 0.0

                        else:

                            # Flat edge at top
                            edge_origin_x = 0.5*moduleR
                            edge_origin_y = moduler
                            edge_dest_x = -0.5*moduleR
                            edge_dest_y = moduler

                        # Flip to bottom half if below x-axis
                        if vertex_y[i] < 0:
                            edge_dest_y *= -1
                            edge_origin_y *= -1

                        # Get edge slope vector
                        edge_slope_x = edge_dest_x - edge_origin_x
                        edge_slope_y = edge_dest_y - edge_origin_y

                        # Project vertices adjacent to the vertex outside the module onto the module edge
                        projection_factor = ((vertex_x[i] - edge_origin_x)*edge_slope_x
                                            + (vertex_y[i] - edge_origin_y)*edge_slope_y)/(edge_slope_x**2 + edge_slope_y**2)

                        proj_x = edge_origin_x + projection_factor*edge_slope_x
                        proj_y = edge_origin_y + projection_factor*edge_slope_y

                        if not isinside[up]:

                            # The next point is outside
                            actual_x[num_vertices] = vertex_x[i]
                            actual_y[num_vertices] = vertex_y[i]
                            actual_x[num_vertices + 1] = proj_x
                            actual_y[num_vertices + 1] = proj_y

                        else:

                            # The previous point was outside
                            actual_x[num_vertices] = proj_x
                            actual_y[num_vertices] = proj_y
                            actual_x[num_vertices + 1] = vertex_x[i]
                            actual_y[num_vertices + 1] = vertex_y[i]

                        num_vertices += 2

                    else:
                        actual_x[num_vertices] = vertex_x[i]
                        actual_y[num_vertices] = vertex_y[i]
                        num_vertices += 1

            else:

                # All 6 inside, just copy the vertices over
                for i in range(0, 6):
                    actual_x[i] = vertex_x[i]
                    actual_y[i] = vertex_y[i]

            ecalPolyVerts.append(np.array([(actual_x[k], actual_y[k]) for k in range(0, 8) if ((actual_x[k] is not None) and (actual_y[k] is not None))]))

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

    return cellPositionMap, np.array(ecalPolyVerts)

# Build the module position map
modulePositionMap = buildModuleMap()

# Build the cell position map
cellPositionMap, ecalPolyVerts = buildCellMap()

# Function to build the cell module position map
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

def buildNeighborMaps():

    # Strategy: Neighbors may include from other modules. All this is precomputed. So we can be wasteful here.
    # Gaps may be nonzero, so we simply apply an annulus requirement (r < point <= r+dr) using total x,y positions
    # relative to the ecal center (cell + module positions). This makes the routine portable to future cell layouts.
    # Note that the module centers already take into account a nonzero gap
    # The number of neighbors is not simple because:Edges, and that module edges have cutoff cells
    #     (NN) Center within [cellr_, 3*cellr_]
    #     (NNN) Center within [3*cellr_, 4.5*cellr_]
    #     Chosen because in ideal case, centers are at 2*cell_ (NN), and at 3*cellR = 3.46*cellr and 4*cellr (NNN)
    NNMap = {}
    NNNMap = {}

    for centerID in cellModulePositionMap:
        NNMap[centerID] = []
        NNNMap[centerID] = []
        centerX = cellModulePositionMap[centerID][0]
        centerY = cellModulePositionMap[centerID][1]

        for probeID in cellModulePositionMap:
            probeX = cellModulePositionMap[probeID][0]
            probeY = cellModulePositionMap[probeID][1]
            dist = math.sqrt((probeX - centerX)**2 + (probeY - centerY)**2)

            if (dist > cellr) and (dist <= 3*cellr):
                NNMap[centerID].append(probeID)

            elif (dist > 3*cellr) and (dist <= 4.5*cellr):
                NNNMap[centerID].append(probeID)

    for centerID in NNMap:
        NNMap[centerID] = np.array(NNMap[centerID])

    for centerID in NNNMap:
        NNNMap[centerID] = np.array(NNNMap[centerID])

    return NNMap, NNNMap

# Build the neighbor maps
NNMap, NNNMap = buildNeighborMaps()

######################################
# Ecal hex readout functions
######################################

def getNN(ID):
    centerID = [centerID_ for centerID_ in NNMap if ecalIDTools.isFlatEcalID(ID, centerID_)][0]
    return np.array([ecalIDTools.EcalID(ID.getLayerID(), probeID.getModuleID(), probeID.getCellID()) for probeID in NNMap[centerID]])

def isNN(ID, probeID):
    for probeID_ in getNN(ID):
        if ecalIDTools.isSameEcalID(probeID, probeID_):
            return True
    return False

def getNNN(ID):
    centerID = [centerID_ for centerID_ in NNNMap if ecalIDTools.isFlatEcalID(ID, centerID_)][0]
    return np.array([ecalIDTools.EcalID(ID.getLayerID(), probeID.getModuleID(), probeID.getCellID()) for probeID in NNNMap[centerID]])

def isNNN(ID, probeID):
    for probeID_ in getNNN(ID):
        if ecalIDTools.isSameEcalID(probeID, probeID_):
            return True
    return False

def getZPosition(layerID):
    return ecalFrontZ + layerZPositions[layerID]

def getCellCenterAbsolute(ID):
    flatID = [flatID_ for flatID_ in cellModulePositionMap if ecalIDTools.isFlatEcalID(ID, flatID_)][0]
    return cellModulePositionMap[flatID]
