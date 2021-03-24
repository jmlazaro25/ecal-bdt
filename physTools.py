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

# For v12 reconstruction
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


###########################
# Miscellaneous
###########################

# Reconstructed energy from sim energy
def recE(siEnergy, layer):
    return ((siEnergy/mipSiEnergy)*layerWeights[layer-1]+siEnergy)*secondOrderEnergyCorrection

# 2D Rotation
def rotate(point,ang):
    ang = np.radians(ang)
    rotM = np.array([[np.cos(ang),-np.sin(ang)],
                    [np.sin(ang), np.cos(ang)]])
    return list(np.dot(rotM,point))

# Project poimt to z_final
def projection(pos_init, mom_init, z_final):
    x_final = pos_init[0] + mom_init[0]/mom_init[2]*(z_final - pos_init[2])
    y_final = pos_init[1] + mom_init[1]/mom_init[2]*(z_final - pos_init[2])
    return (x_final, y_final)

# List of projected (x,y)s at each ECal layer
def layerIntercepts(pos,mom,layerZs=ecal_layerZs):
    return [projection(pos,mom,z) for z in layerZs]

# Magnitude of whatever
def mag(iterable):
    return np.sqrt(sum([x**2 for x in iterable]))

# Distance detween points
def dist(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    return mag(p1 - p2)

# Angle with Z
def angle(vec):
    return np.degrees(np.arccos(vec[2]/mag(vec)))


###########################
# Get e/gamma SP hit info
###########################

# Electron Target SP hit
def electronTargetSPHit(targetSPHits):

    hitOfInt, pmax = None, 0
    for hit in targetSPHits:

        if hit.getPosition()[2] > sp_thickness + 1 or\
                hit.getMomentum()[2] <= 0 or\
                hit.getTrackID() != 1 or\
                hit.getPdgID() != 11:
            continue

        if mag(hit.getMomentum()) > pmax:
            hitOfInt = hit
            pmax = mag(hitOfInt.getMomentum())

    return hitOfInt

# Photon Target SP hit
def gammaTargetSPHit(targetSPHits):

    hitOfInt, pmax = None, 0
    for hit in targetSPHits:

        if hit.getPosition()[2] > sp_thickness  + 1 or\
                hit.getMomentum()[2] <= 0 or\
                not (hit.getPdgID() in [-22,22]):
            continue

        if mag(hit.getMomentum()) > pmax:
            hitOfInt = hit
            pmax = mag(hitOfInt.getMomentum())

    return hitOfInt

# Electron ECal SP hit
def electronEcalSPHit(ecalSPHits):

    hitOfInt, pmax = None, 0
    for hit in ecalSPHits:

        if hit.getPosition()[2] > ecal_front_z + sp_thickness or\
                hit.getMomentum()[2] <= 0 or\
                hit.getTrackID() != 1 or\
                hit.getPdgID() != 11:
            continue

        if mag(hit.getMomentum()) > pmax:
            hitOfInt = hit
            pmax = mag(hitOfInt.getMomentum())

    return hitOfInt

# Photon ECal SP hit
def gammaEcalSPHit(ecalSPHits):

    hitOfInt, pmax = None, 0
    for hit in ecalSPHits:

        if hit.getPosition()[2] > ecal_front_z + sp_thickness or\
                hit.getMomentum()[2] <= 0 or\
                not (hit.getPdgID() in [-22,22]):
            continue

        if mag(hit.getMomentum()) > pmax:
            hitOfInt = hit
            pmax = mag(hitOfInt.getMomentum())

    return hitOfInt

