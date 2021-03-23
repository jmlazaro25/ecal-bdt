import numpy as np
# all space values in mm unless otherwise noted

#gdml values
ecal_front_z = 240.5
sp_thickness = 0.001 
clearance = 0.001
ECal_dz = 449.2
ecal_envelope_z = ECal_dz + 1
sp_ecal_front_z = ecal_front_z + (ecal_envelope_z - ECal_dz)/2 - sp_thickness/2 + clearance

sin60 = np.sin(np.radians(60))
module_radius = 85. # dist form center to midpoint of side
module_side = module_radius/sin60
module_gap = 1.5 # space between sides of side-by-side mods

ecal_layerZs = ecal_front_z + np.array([7.850,   13.300,  26.400,  33.500,  47.950,
                                        56.550,  72.250,  81.350,  97.050,  106.150,
                                        121.850, 130.950, 146.650, 155.750, 171.450,
                                        180.550, 196.250, 205.350, 221.050, 230.150,
                                        245.850, 254.950, 270.650, 279.750, 298.950,
                                        311.550, 330.750, 343.350, 362.550, 375.150,
                                        394.350, 406.950, 426.150, 438.750        ])

# For reconstruction for v12
mipSiEnergy = 0.130 # MeV
secondOrderEnergyCorrection = 4000./4010.
layerWeights = [
            1.675, 2.724, 4.398, 6.039, 7.696, 9.077, 9.630, 9.630, 9.630, 9.630, 9.630,
            9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630, 9.630,
            9.630, 13.497, 17.364, 17.364, 17.364, 17.364, 17.364, 17.364, 17.364, 17.364,
            17.364, 8.990
            ]
ecal_zs_round = [round(z) for z in ecal_layerZs]
ecal_rz2layer = {}
for rz, i in zip(ecal_zs_round,range(1,35)):
    ecal_rz2layer[rz] = i

def recE(siEnergy, layer):
    return ((siEnergy/mipSiEnergy)*layerWeights[layer-1]+siEnergy)*secondOrderEnergyCorrection

# Borrowed/editted from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

# Given three colinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False

def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  

    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):

        # Clockwise orientation 
        return 1
    elif (val < 0):

        # Counterclockwise orientation 
        return 2
    else:

        # Colinear orientation 
        return 0

# The main function that returns true if  
# the line segment 'mseg' and 'rseg' intersect. 
def doIntersect(mseg,rseg):

    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(mseg[0], mseg[1], rseg[0])
    o2 = orientation(mseg[0], mseg[1], rseg[1])
    o3 = orientation(rseg[0], rseg[1], mseg[0])
    o4 = orientation(rseg[0], rseg[1], mseg[1])

    # General case 
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases 

    # mseg[0] , mseg[1] and rseg[0] are colinear and rseg[0] lies on segment mseg[0]mseg[1] 
    if ((o1 == 0) and onSegment(mseg[0], rseg[0], mseg[1])):
        return True

    # mseg[0] , mseg[1] and rseg[1] are colinear and rseg[1] lies on segment mseg[0]mseg[1] 
    if ((o2 == 0) and onSegment(mseg[0], rseg[1], mseg[1])):
        return True

    # rseg[0] , rseg[1] and mseg[0] are colinear and mseg[0] lies on segment rseg[0]rseg[1] 
    if ((o3 == 0) and onSegment(rseg[0], mseg[0], rseg[1])):
        return True

    # rseg[0] , rseg[1] and mseg[1] are colinear and mseg[1] lies on segment rseg[0]rseg[1] 
    if ((o4 == 0) and onSegment(rseg[0], mseg[1], rseg[1])):
        return True

    # If none of the cases 
    return False

# Borrow END

class module:

    def __init__(self, center, layer, config='full_90'):
        self.layer = layer
        self.cx, self.cy = center[0], center[1]
        self.cz = ecal_layerZs[self.layer - 1]
        # to use v9 (or other non standard det, define layerZs in importing script

        shape, angle = config.split('_')[0], float(config.split('_')[1])

        ###########################
        # Draw
        ########################### 
        # Base is full module with point up
        self.corners = [     # from top corner; clockwise
                        [0,               module_side  ],
                        [module_radius,   module_side/2],
                        [module_radius,  -module_side/2],
                        [0,              -module_side  ],
                        [-module_radius, -module_side/2],
                        [-module_radius,  module_side/2]
                        ]
        # For five, pop bottom point
        if shape == 'five':
            self.corners.pop(3)
        # For a half, pop left half points
        if shape == 'half':
            self.corners.pop(4); self.corners.pop(4)
        # For three, pop all but top 3
        if shape == 'three':
            self.corners.pop(2); self.corners.pop(2); self.corners.pop(2)
        # Others going to take more; make when/if needed

        ###########################
        # Rotate
        ###########################
        if angle != 0:
            self.corners = [rotate(corner,angle) for corner in self.corners]

        ###########################
        # Move into posittion
        ###########################
        for corner in self.corners:
            corner[0] += self.cx
            corner[1] += self.cy

        ###########################
        # Connect the dots
        ###########################
        self.sides = [[self.corners[k],self.corners[k+1]] for k in range(len(self.corners)-1)]
        self.sides.append([self.corners[-1],self.corners[0]]) # connect back to first corner

    def contains(self, xy):
        #if xyz[2] != self.cz: return False # don't bother if not at same z removing 3D-ness
        ray = [(xy[0],xy[1]),(xy[0]+500,xy[1]+500)] # draw a ray out from xyz
        crossings = 0
        for side in self.sides:
            if doIntersect(ray,side): crossings += 1
        if crossings %2 == 0: return False
        return True

class detector:

    def __init__(self, config='v12'):
        self.layers = {}

        ###########################
        # Init Centers
        ###########################
        # Optimal 7 module flower (config_0) is base (but not default)
        self.centers = [
                (0,0),                           # center to top then clockwise
                (-module_gap/2 - module_radius, (2*module_radius+module_gap)*sin60),
                (module_gap/2 + module_radius, (2*module_radius+module_gap)*sin60),
                (2*module_radius + module_gap, 0),
                (module_gap/2 + module_radius, -(2*module_radius+module_gap)*sin60),
                (-module_gap/2 - module_radius, -(2*module_radius+module_gap)*sin60),
                (-2*module_radius - module_gap, 0)
                ]

        # Add and rotate centers as needed for other test configs
        if config == 'config_1' or\
                config == 'config_2' or config == 'xLR' or\
                config == 'config_6' or config == 'square':
            self.centers.append((-3*module_radius-3*module_gap/2,\
                                -(2*module_radius+module_gap)*sin60))
            self.centers.append((-3*module_radius-3*module_gap/2,\
                                (2*module_radius+module_gap)*sin60))
            if config == 'config_2' or config == 'xLR' or\
                    config == 'config_6' or config == 'square':
                self.centers.append((3*module_radius+3*module_gap/2,\
                                -(2*module_radius+module_gap)*sin60))
                self.centers.append((3*module_radius+3*module_gap/2,\
                                (2*module_radius+module_gap)*sin60))
                if config == 'config_6' or config == 'square':
                    # Top left to bottom right; like en reading
                    self.centers.append((-2*module_radius-module_gap,\
                                (4*module_radius+2*module_gap)*sin60))
                    self.centers.append((0 ,\
                                (4*module_radius+2*module_gap)*sin60))
                    self.centers.append(( 2*module_radius+module_gap,\
                                (4*module_radius+2*module_gap)*sin60))
                    self.centers.append((-2*module_radius-module_gap,\
                                (-4*module_radius-2*module_gap)*sin60))
                    self.centers.append((0 ,\
                                (-4*module_radius-2*module_gap)*sin60))
                    self.centers.append(( 2*module_radius+module_gap,\
                                (-4*module_radius-2*module_gap)*sin60))
        if config == 'v12' or config == 'config_3':
            self.centers = [rotate(center,90) for center in self.centers]

        ###########################
        # Draw modules @ centers
        ###########################
        for layer in range(1,len(ecal_layerZs)+1):
            self.layers[layer] = []
            mod_count = -1 
            for center in self.centers:
                mod_count += 1
                if config == 'v12' or config == 'config_3':
                    self.layers[layer].append(module(center, layer, 'full_90'))
                if config == 'config_0':
                    self.layers[layer].append(module(center, layer, 'full_0'))
                if config == 'config_1'  or\
                    config == 'config_2' or config == 'xLR' or\
                    config == 'config_6' or config == 'square':
                    if mod_count < 7:
                        self.layers[layer].append(module(center, layer, 'full_0'))
                    elif mod_count < 9:
                        self.layers[layer].append(module(center, layer, 'half_0'))
                    elif mod_count < 11:
                        self.layers[layer].append(module(center, layer, 'half_180'))
                    elif mod_count < 14:
                        self.layers[layer].append(module(center, layer, 'three_180'))
                    else:
                        self.layers[layer].append(module(center, layer, 'three_0'))

    def plotLayer(self,layer=1):
        import matplotlib.pyplot as plt
        for mod in self.layers[layer]:
            for side in mod.sides:
                xs = [corner[0] for corner in side]
                ys = [corner[1] for corner in side]
                plt.plot(xs,ys,color='k')
            ax = plt.gca().invert_xaxis()
            plt.xlabel('x (mm)')
            plt.ylabel('y (mm)')

def rotate(point,ang):
    ang = np.radians(ang)
    rotM = np.array([[np.cos(ang),-np.sin(ang)],
                    [np.sin(ang), np.cos(ang)]])
    return list(np.dot(rotM,point))

def projection(pos_init, mom_init, z_final):
    x_final = pos_init[0] + mom_init[0]/mom_init[2]*(z_final - pos_init[2])
    y_final = pos_init[1] + mom_init[1]/mom_init[2]*(z_final - pos_init[2])
    return (x_final, y_final)

def layerIntercepts(pos,mom,layerZs=ecal_layerZs):
    return [projection(pos,mom,z) for z in layerZs]

def mag(iterable):
    return np.sqrt(sum([x**2 for x in iterable])) 

def dist(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    return mag(p1 - p2)

def angle(vec):
    return np.degrees(np.arccos(vec[2]/mag(vec)))

def electronSPHits(ecalSPHits, targetSPHits):
    
    ecalSPHit, targetSPHit = None, None

    # Find ecal SP hit of recoil electron
    pmax = 0
    for hit in ecalSPHits:

        if hit.getPosition()[2] > ecal_front_z + sp_thickness or\
                hit.getMomentum()[2] <= 0 or\
                hit.getTrackID() != 1:
            continue

        if mag(hit.getMomentum()) > pmax:
            ecalSPHit = hit
            pmax = mag(ecalSPHit.getMomentum())

    # Find target SP hit of recoil electron
    pmax = 0
    for hit in targetSPHits:
    
        if hit.getPosition()[2] > sp_thickness or\
                hit.getMomentum()[2] <= 0 or\
                hit.getTrackID() != 1:
            continue

        if mag(hit.getMomentum()) > pmax:
            targetSPHit = hit
            pmax = mag(targetSPHit.getMomentum())

    return ecalSPHit, targetSPHit

def elec_gamma_ecalSPHits(ecalSPHits):

    eSPHit = electronEcalSPHit(ecalSPHits)
    gSPHit = gammaEcalSPHit(ecalSPHits)

    return eSPHit, gSPHit

def electronEcalSPHit(ecalSPHits):
    
    eSPHit = None
    pmax = 0
    for hit in ecalSPHits:

        if hit.getPosition()[2] > ecal_front_z + sp_thickness + 0.5 or\
                hit.getMomentum()[2] <= 0 or\
                hit.getTrackID() != 1 or\
                hit.getPdgID() != 11:
            continue

        if mag(hit.getMomentum()) > pmax:
            eSPHit = hit          
            pmax = mag(eSPHit.getMomentum())

    return eSPHit

def gammaEcalSPHit(ecalSPHits):
    
    gSPHit = None
    pmax = 0
    for hit in ecalSPHits:

        if hit.getPosition()[2] > ecal_front_z + sp_thickness + 0.5 or\
                hit.getMomentum()[2] <= 0 or\
                not (hit.getPdgID() in [-22,22]):
            continue

        if mag(hit.getMomentum()) > pmax:
            gSPHit = hit
            pmax = mag(gSPHit.getMomentum())

    return gSPHit
