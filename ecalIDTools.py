LAYER_MASK = 0x3F
LAYER_SHIFT = 17
MODULE_MASK = 0x1F
MODULE_SHIFT = 12
CELL_MASK = 0xFFF
CELL_SHIFT = 0

######################################
# Hit information from hitID
######################################

# Function to extract the layer from a hitID
def hitIDToLayerID(hitID):
    return (hitID>>LAYER_SHIFT)&LAYER_MASK

# Function to extract the module from a hitID
def hitIDToModuleID(hitID):
    return (hitID>>MODULE_SHIFT)&MODULE_MASK

# Function to extract the cell from a hitID
def hitIDToCellID(hitID):
    return (hitID>>CELL_SHIFT)&CELL_MASK

################################################
# Naive implementation of EcalID class
################################################

class EcalID:

    def __init__(self, layerID, moduleID, cellID):

        self.layerID = layerID
        self.moduleID = moduleID
        self.cellID = cellID

    def setLayerID(self, newLayerID):
        self.layerID = newLayerID

    def getLayerID(self):
        return self.layerID

    def setModuleID(self, newModuleID):
        self.moduleID = newModuleID

    def getModuleID(self):
        return self.moduleID

    def setCellID(self, newCellID):
        self.cell = newCellID

    def getCellID(self):
        return self.cellID

def emptyEcalID():
    return EcalID(None, None, None)
