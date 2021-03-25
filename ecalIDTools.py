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
def IDToLayer(hitID):
    return (hitID>>LAYER_SHIFT)&LAYER_MASK

# Function to extract the module from a hitID
def IDToModule(hitID):
    return (hitID>>MODULE_SHIFT)&MODULE_MASK

# Function to extract the cell from a hitID
def IDToCell(hitID):
    return (hitID>>CELL_SHIFT)&CELL_MASK

################################################
# Naive implementation of EcalID class
################################################

class EcalID:

    def __init__(self, layer, module, cell):

        self.layer = layer
        self.module = module
        self.cell = cell

    def setLayer(self, newLayer):
        self.layer = newLayer

    def getLayer(self):
        return self.layer

    def setModule(self, newModule):
        self.module = newModule

    def getModule(self):
        return self.module

    def setCell(self, newCell):
        self.cell = newCell

    def getCell(self):
        return self.cell

def emptyEcalID():
    return EcalID(None, None, None)
