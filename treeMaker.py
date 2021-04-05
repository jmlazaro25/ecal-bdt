import os
import math
import ROOT as r
import numpy as np
from mods import ROOTmanager as manager
from mods import physTools, mipTracking
r.gSystem.Load('/nfs/slac/g/ldmx/users/aechavez/ldmx-sw-v2.3.0-w-container/ldmx-sw/install/lib/libEvent.so')
#r.gSystem.Load('/home/jmlazaro/research/ldmx-sw/install/lib/libEvent.so')

# TreeModel to build here
branches_info = {
        # Base variables
        'nReadoutHits':              {'rtype': int,   'default': 0 },
        'summedDet':                 {'rtype': float, 'default': 0.},
        'summedTightIso':            {'rtype': float, 'default': 0.},
        'maxCellDep':                {'rtype': float, 'default': 0.},
        'showerRMS':                 {'rtype': float, 'default': 0.},
        'xStd':                      {'rtype': float, 'default': 0.},
        'yStd':                      {'rtype': float, 'default': 0.},
        'avgLayerHit':               {'rtype': float, 'default': 0.},
        'stdLayerHit':               {'rtype': float, 'default': 0.},
        'deepestLayerHit':           {'rtype': int,   'default': 0 },
        'ecalBackEnergy':            {'rtype': float, 'default': 0.},
        # Longitudinal segment variables
        'energy_s1':                 {'rtype': float, 'default': 0.},
        'energy_s2':                 {'rtype': float, 'default': 0.},
        'energy_s3':                 {'rtype': float, 'default': 0.},
        'nHits_s1':                  {'rtype': int, 'default': 0},
        'nHits_s2':                  {'rtype': int, 'default': 0},
        'nHits_s3':                  {'rtype': int, 'default': 0},
        'xMean_s1':                  {'rtype': float, 'default': 0.},
        'xMean_s2':                  {'rtype': float, 'default': 0.},
        'xMean_s3':                  {'rtype': float, 'default': 0.},
        'yMean_s1':                  {'rtype': float, 'default': 0.},
        'yMean_s2':                  {'rtype': float, 'default': 0.},
        'yMean_s3':                  {'rtype': float, 'default': 0.},
        'layerMean_s1':              {'rtype': float, 'default': 0.},
        'layerMean_s2':              {'rtype': float, 'default': 0.},
        'layerMean_s3':              {'rtype': float, 'default': 0.},
        'xStd_s1':                   {'rtype': float, 'default': 0.},
        'xStd_s2':                   {'rtype': float, 'default': 0.},
        'xStd_s3':                   {'rtype': float, 'default': 0.},
        'yStd_s1':                   {'rtype': float, 'default': 0.},
        'yStd_s2':                   {'rtype': float, 'default': 0.},
        'yStd_s3':                   {'rtype': float, 'default': 0.},
        'layerStd_s1':               {'rtype': float, 'default': 0.},
        'layerStd_s2':               {'rtype': float, 'default': 0.},
        'layerStd_s3':               {'rtype': float, 'default': 0.},
        # Electron radius of containment variables
        'eContEnergy_x1_s1':         {'rtype': float, 'default': 0.},
        'eContEnergy_x2_s1':         {'rtype': float, 'default': 0.},
        'eContEnergy_x3_s1':         {'rtype': float, 'default': 0.},
        'eContEnergy_x4_s1':         {'rtype': float, 'default': 0.},
        'eContEnergy_x5_s1':         {'rtype': float, 'default': 0.},
        'eContEnergy_x1_s2':         {'rtype': float, 'default': 0.},
        'eContEnergy_x2_s2':         {'rtype': float, 'default': 0.},
        'eContEnergy_x3_s2':         {'rtype': float, 'default': 0.},
        'eContEnergy_x4_s2':         {'rtype': float, 'default': 0.},
        'eContEnergy_x5_s2':         {'rtype': float, 'default': 0.},
        'eContEnergy_x1_s3':         {'rtype': float, 'default': 0.},
        'eContEnergy_x2_s3':         {'rtype': float, 'default': 0.},
        'eContEnergy_x3_s3':         {'rtype': float, 'default': 0.},
        'eContEnergy_x4_s3':         {'rtype': float, 'default': 0.},
        'eContEnergy_x5_s3':         {'rtype': float, 'default': 0.},
        'eContNHits_x1_s1':          {'rtype': int, 'default': 0},
        'eContNHits_x2_s1':          {'rtype': int, 'default': 0},
        'eContNHits_x3_s1':          {'rtype': int, 'default': 0},
        'eContNHits_x4_s1':          {'rtype': int, 'default': 0},
        'eContNHits_x5_s1':          {'rtype': int, 'default': 0},
        'eContNHits_x1_s2':          {'rtype': int, 'default': 0},
        'eContNHits_x2_s2':          {'rtype': int, 'default': 0},
        'eContNHits_x3_s2':          {'rtype': int, 'default': 0},
        'eContNHits_x4_s2':          {'rtype': int, 'default': 0},
        'eContNHits_x5_s2':          {'rtype': int, 'default': 0},
        'eContNHits_x1_s3':          {'rtype': int, 'default': 0},
        'eContNHits_x2_s3':          {'rtype': int, 'default': 0},
        'eContNHits_x3_s3':          {'rtype': int, 'default': 0},
        'eContNHits_x4_s3':          {'rtype': int, 'default': 0},
        'eContNHits_x5_s3':          {'rtype': int, 'default': 0},
        'eContXMean_x1_s1':          {'rtype': float, 'default': 0.},
        'eContXMean_x2_s1':          {'rtype': float, 'default': 0.},
        'eContXMean_x3_s1':          {'rtype': float, 'default': 0.},
        'eContXMean_x4_s1':          {'rtype': float, 'default': 0.},
        'eContXMean_x5_s1':          {'rtype': float, 'default': 0.},
        'eContXMean_x1_s2':          {'rtype': float, 'default': 0.},
        'eContXMean_x2_s2':          {'rtype': float, 'default': 0.},
        'eContXMean_x3_s2':          {'rtype': float, 'default': 0.},
        'eContXMean_x4_s2':          {'rtype': float, 'default': 0.},
        'eContXMean_x5_s2':          {'rtype': float, 'default': 0.},
        'eContXMean_x1_s3':          {'rtype': float, 'default': 0.},
        'eContXMean_x2_s3':          {'rtype': float, 'default': 0.},
        'eContXMean_x3_s3':          {'rtype': float, 'default': 0.},
        'eContXMean_x4_s3':          {'rtype': float, 'default': 0.},
        'eContXMean_x5_s3':          {'rtype': float, 'default': 0.},
        'eContYMean_x1_s1':          {'rtype': float, 'default': 0.},
        'eContYMean_x2_s1':          {'rtype': float, 'default': 0.},
        'eContYMean_x3_s1':          {'rtype': float, 'default': 0.},
        'eContYMean_x4_s1':          {'rtype': float, 'default': 0.},
        'eContYMean_x5_s1':          {'rtype': float, 'default': 0.},
        'eContYMean_x1_s2':          {'rtype': float, 'default': 0.},
        'eContYMean_x2_s2':          {'rtype': float, 'default': 0.},
        'eContYMean_x3_s2':          {'rtype': float, 'default': 0.},
        'eContYMean_x4_s2':          {'rtype': float, 'default': 0.},
        'eContYMean_x5_s2':          {'rtype': float, 'default': 0.},
        'eContYMean_x1_s3':          {'rtype': float, 'default': 0.},
        'eContYMean_x2_s3':          {'rtype': float, 'default': 0.},
        'eContYMean_x3_s3':          {'rtype': float, 'default': 0.},
        'eContYMean_x4_s3':          {'rtype': float, 'default': 0.},
        'eContYMean_x5_s3':          {'rtype': float, 'default': 0.},
        'eContLayerMean_x1_s1':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x2_s1':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x3_s1':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x4_s1':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x5_s1':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x1_s2':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x2_s2':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x3_s2':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x4_s2':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x5_s2':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x1_s3':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x2_s3':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x3_s3':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x4_s3':      {'rtype': float, 'default': 0.},
        'eContLayerMean_x5_s3':      {'rtype': float, 'default': 0.},
        'eContXStd_x1_s1':           {'rtype': float, 'default': 0.},
        'eContXStd_x2_s1':           {'rtype': float, 'default': 0.},
        'eContXStd_x3_s1':           {'rtype': float, 'default': 0.},
        'eContXStd_x4_s1':           {'rtype': float, 'default': 0.},
        'eContXStd_x5_s1':           {'rtype': float, 'default': 0.},
        'eContXStd_x1_s2':           {'rtype': float, 'default': 0.},
        'eContXStd_x2_s2':           {'rtype': float, 'default': 0.},
        'eContXStd_x3_s2':           {'rtype': float, 'default': 0.},
        'eContXStd_x4_s2':           {'rtype': float, 'default': 0.},
        'eContXStd_x5_s2':           {'rtype': float, 'default': 0.},
        'eContXStd_x1_s3':           {'rtype': float, 'default': 0.},
        'eContXStd_x2_s3':           {'rtype': float, 'default': 0.},
        'eContXStd_x3_s3':           {'rtype': float, 'default': 0.},
        'eContXStd_x4_s3':           {'rtype': float, 'default': 0.},
        'eContXStd_x5_s3':           {'rtype': float, 'default': 0.},
        'eContYStd_x1_s1':           {'rtype': float, 'default': 0.},
        'eContYStd_x2_s1':           {'rtype': float, 'default': 0.},
        'eContYStd_x3_s1':           {'rtype': float, 'default': 0.},
        'eContYStd_x4_s1':           {'rtype': float, 'default': 0.},
        'eContYStd_x5_s1':           {'rtype': float, 'default': 0.},
        'eContYStd_x1_s2':           {'rtype': float, 'default': 0.},
        'eContYStd_x2_s2':           {'rtype': float, 'default': 0.},
        'eContYStd_x3_s2':           {'rtype': float, 'default': 0.},
        'eContYStd_x4_s2':           {'rtype': float, 'default': 0.},
        'eContYStd_x5_s2':           {'rtype': float, 'default': 0.},
        'eContYStd_x1_s3':           {'rtype': float, 'default': 0.},
        'eContYStd_x2_s3':           {'rtype': float, 'default': 0.},
        'eContYStd_x3_s3':           {'rtype': float, 'default': 0.},
        'eContYStd_x4_s3':           {'rtype': float, 'default': 0.},
        'eContYStd_x5_s3':           {'rtype': float, 'default': 0.},
        'eContLayerStd_x1_s1':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x2_s1':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x3_s1':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x4_s1':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x5_s1':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x1_s2':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x2_s2':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x3_s2':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x4_s2':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x5_s2':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x1_s3':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x2_s3':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x3_s3':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x4_s3':       {'rtype': float, 'default': 0.},
        'eContLayerStd_x5_s3':       {'rtype': float, 'default': 0.},
        # Photon radius of containment variables
        'gContEnergy_x1_s1':         {'rtype': float, 'default': 0.},
        'gContEnergy_x2_s1':         {'rtype': float, 'default': 0.},
        'gContEnergy_x3_s1':         {'rtype': float, 'default': 0.},
        'gContEnergy_x4_s1':         {'rtype': float, 'default': 0.},
        'gContEnergy_x5_s1':         {'rtype': float, 'default': 0.},
        'gContEnergy_x1_s2':         {'rtype': float, 'default': 0.},
        'gContEnergy_x2_s2':         {'rtype': float, 'default': 0.},
        'gContEnergy_x3_s2':         {'rtype': float, 'default': 0.},
        'gContEnergy_x4_s2':         {'rtype': float, 'default': 0.},
        'gContEnergy_x5_s2':         {'rtype': float, 'default': 0.},
        'gContEnergy_x1_s3':         {'rtype': float, 'default': 0.},
        'gContEnergy_x2_s3':         {'rtype': float, 'default': 0.},
        'gContEnergy_x3_s3':         {'rtype': float, 'default': 0.},
        'gContEnergy_x4_s3':         {'rtype': float, 'default': 0.},
        'gContEnergy_x5_s3':         {'rtype': float, 'default': 0.},
        'gContNHits_x1_s1':          {'rtype': int, 'default': 0},
        'gContNHits_x2_s1':          {'rtype': int, 'default': 0},
        'gContNHits_x3_s1':          {'rtype': int, 'default': 0},
        'gContNHits_x4_s1':          {'rtype': int, 'default': 0},
        'gContNHits_x5_s1':          {'rtype': int, 'default': 0},
        'gContNHits_x1_s2':          {'rtype': int, 'default': 0},
        'gContNHits_x2_s2':          {'rtype': int, 'default': 0},
        'gContNHits_x3_s2':          {'rtype': int, 'default': 0},
        'gContNHits_x4_s2':          {'rtype': int, 'default': 0},
        'gContNHits_x5_s2':          {'rtype': int, 'default': 0},
        'gContNHits_x1_s3':          {'rtype': int, 'default': 0},
        'gContNHits_x2_s3':          {'rtype': int, 'default': 0},
        'gContNHits_x3_s3':          {'rtype': int, 'default': 0},
        'gContNHits_x4_s3':          {'rtype': int, 'default': 0},
        'gContNHits_x5_s3':          {'rtype': int, 'default': 0},
        'gContXMean_x1_s1':          {'rtype': float, 'default': 0.},
        'gContXMean_x2_s1':          {'rtype': float, 'default': 0.},
        'gContXMean_x3_s1':          {'rtype': float, 'default': 0.},
        'gContXMean_x4_s1':          {'rtype': float, 'default': 0.},
        'gContXMean_x5_s1':          {'rtype': float, 'default': 0.},
        'gContXMean_x1_s2':          {'rtype': float, 'default': 0.},
        'gContXMean_x2_s2':          {'rtype': float, 'default': 0.},
        'gContXMean_x3_s2':          {'rtype': float, 'default': 0.},
        'gContXMean_x4_s2':          {'rtype': float, 'default': 0.},
        'gContXMean_x5_s2':          {'rtype': float, 'default': 0.},
        'gContXMean_x1_s3':          {'rtype': float, 'default': 0.},
        'gContXMean_x2_s3':          {'rtype': float, 'default': 0.},
        'gContXMean_x3_s3':          {'rtype': float, 'default': 0.},
        'gContXMean_x4_s3':          {'rtype': float, 'default': 0.},
        'gContXMean_x5_s3':          {'rtype': float, 'default': 0.},
        'gContYMean_x1_s1':          {'rtype': float, 'default': 0.},
        'gContYMean_x2_s1':          {'rtype': float, 'default': 0.},
        'gContYMean_x3_s1':          {'rtype': float, 'default': 0.},
        'gContYMean_x4_s1':          {'rtype': float, 'default': 0.},
        'gContYMean_x5_s1':          {'rtype': float, 'default': 0.},
        'gContYMean_x1_s2':          {'rtype': float, 'default': 0.},
        'gContYMean_x2_s2':          {'rtype': float, 'default': 0.},
        'gContYMean_x3_s2':          {'rtype': float, 'default': 0.},
        'gContYMean_x4_s2':          {'rtype': float, 'default': 0.},
        'gContYMean_x5_s2':          {'rtype': float, 'default': 0.},
        'gContYMean_x1_s3':          {'rtype': float, 'default': 0.},
        'gContYMean_x2_s3':          {'rtype': float, 'default': 0.},
        'gContYMean_x3_s3':          {'rtype': float, 'default': 0.},
        'gContYMean_x4_s3':          {'rtype': float, 'default': 0.},
        'gContYMean_x5_s3':          {'rtype': float, 'default': 0.},
        'gContLayerMean_x1_s1':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x2_s1':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x3_s1':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x4_s1':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x5_s1':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x1_s2':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x2_s2':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x3_s2':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x4_s2':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x5_s2':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x1_s3':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x2_s3':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x3_s3':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x4_s3':      {'rtype': float, 'default': 0.},
        'gContLayerMean_x5_s3':      {'rtype': float, 'default': 0.},
        'gContXStd_x1_s1':           {'rtype': float, 'default': 0.},
        'gContXStd_x2_s1':           {'rtype': float, 'default': 0.},
        'gContXStd_x3_s1':           {'rtype': float, 'default': 0.},
        'gContXStd_x4_s1':           {'rtype': float, 'default': 0.},
        'gContXStd_x5_s1':           {'rtype': float, 'default': 0.},
        'gContXStd_x1_s2':           {'rtype': float, 'default': 0.},
        'gContXStd_x2_s2':           {'rtype': float, 'default': 0.},
        'gContXStd_x3_s2':           {'rtype': float, 'default': 0.},
        'gContXStd_x4_s2':           {'rtype': float, 'default': 0.},
        'gContXStd_x5_s2':           {'rtype': float, 'default': 0.},
        'gContXStd_x1_s3':           {'rtype': float, 'default': 0.},
        'gContXStd_x2_s3':           {'rtype': float, 'default': 0.},
        'gContXStd_x3_s3':           {'rtype': float, 'default': 0.},
        'gContXStd_x4_s3':           {'rtype': float, 'default': 0.},
        'gContXStd_x5_s3':           {'rtype': float, 'default': 0.},
        'gContYStd_x1_s1':           {'rtype': float, 'default': 0.},
        'gContYStd_x2_s1':           {'rtype': float, 'default': 0.},
        'gContYStd_x3_s1':           {'rtype': float, 'default': 0.},
        'gContYStd_x4_s1':           {'rtype': float, 'default': 0.},
        'gContYStd_x5_s1':           {'rtype': float, 'default': 0.},
        'gContYStd_x1_s2':           {'rtype': float, 'default': 0.},
        'gContYStd_x2_s2':           {'rtype': float, 'default': 0.},
        'gContYStd_x3_s2':           {'rtype': float, 'default': 0.},
        'gContYStd_x4_s2':           {'rtype': float, 'default': 0.},
        'gContYStd_x5_s2':           {'rtype': float, 'default': 0.},
        'gContYStd_x1_s3':           {'rtype': float, 'default': 0.},
        'gContYStd_x2_s3':           {'rtype': float, 'default': 0.},
        'gContYStd_x3_s3':           {'rtype': float, 'default': 0.},
        'gContYStd_x4_s3':           {'rtype': float, 'default': 0.},
        'gContYStd_x5_s3':           {'rtype': float, 'default': 0.},
        'gContLayerStd_x1_s1':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x2_s1':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x3_s1':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x4_s1':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x5_s1':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x1_s2':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x2_s2':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x3_s2':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x4_s2':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x5_s2':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x1_s3':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x2_s3':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x3_s3':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x4_s3':       {'rtype': float, 'default': 0.},
        'gContLayerStd_x5_s3':       {'rtype': float, 'default': 0.},
        # Outside radius of containment variables
        'oContEnergy_x1_s1':         {'rtype': float, 'default': 0.},
        'oContEnergy_x2_s1':         {'rtype': float, 'default': 0.},
        'oContEnergy_x3_s1':         {'rtype': float, 'default': 0.},
        'oContEnergy_x4_s1':         {'rtype': float, 'default': 0.},
        'oContEnergy_x5_s1':         {'rtype': float, 'default': 0.},
        'oContEnergy_x1_s2':         {'rtype': float, 'default': 0.},
        'oContEnergy_x2_s2':         {'rtype': float, 'default': 0.},
        'oContEnergy_x3_s2':         {'rtype': float, 'default': 0.},
        'oContEnergy_x4_s2':         {'rtype': float, 'default': 0.},
        'oContEnergy_x5_s2':         {'rtype': float, 'default': 0.},
        'oContEnergy_x1_s3':         {'rtype': float, 'default': 0.},
        'oContEnergy_x2_s3':         {'rtype': float, 'default': 0.},
        'oContEnergy_x3_s3':         {'rtype': float, 'default': 0.},
        'oContEnergy_x4_s3':         {'rtype': float, 'default': 0.},
        'oContEnergy_x5_s3':         {'rtype': float, 'default': 0.},
        'oContNHits_x1_s1':          {'rtype': int, 'default': 0},
        'oContNHits_x2_s1':          {'rtype': int, 'default': 0},
        'oContNHits_x3_s1':          {'rtype': int, 'default': 0},
        'oContNHits_x4_s1':          {'rtype': int, 'default': 0},
        'oContNHits_x5_s1':          {'rtype': int, 'default': 0},
        'oContNHits_x1_s2':          {'rtype': int, 'default': 0},
        'oContNHits_x2_s2':          {'rtype': int, 'default': 0},
        'oContNHits_x3_s2':          {'rtype': int, 'default': 0},
        'oContNHits_x4_s2':          {'rtype': int, 'default': 0},
        'oContNHits_x5_s2':          {'rtype': int, 'default': 0},
        'oContNHits_x1_s3':          {'rtype': int, 'default': 0},
        'oContNHits_x2_s3':          {'rtype': int, 'default': 0},
        'oContNHits_x3_s3':          {'rtype': int, 'default': 0},
        'oContNHits_x4_s3':          {'rtype': int, 'default': 0},
        'oContNHits_x5_s3':          {'rtype': int, 'default': 0},
        'oContXMean_x1_s1':          {'rtype': float, 'default': 0.},
        'oContXMean_x2_s1':          {'rtype': float, 'default': 0.},
        'oContXMean_x3_s1':          {'rtype': float, 'default': 0.},
        'oContXMean_x4_s1':          {'rtype': float, 'default': 0.},
        'oContXMean_x5_s1':          {'rtype': float, 'default': 0.},
        'oContXMean_x1_s2':          {'rtype': float, 'default': 0.},
        'oContXMean_x2_s2':          {'rtype': float, 'default': 0.},
        'oContXMean_x3_s2':          {'rtype': float, 'default': 0.},
        'oContXMean_x4_s2':          {'rtype': float, 'default': 0.},
        'oContXMean_x5_s2':          {'rtype': float, 'default': 0.},
        'oContXMean_x1_s3':          {'rtype': float, 'default': 0.},
        'oContXMean_x2_s3':          {'rtype': float, 'default': 0.},
        'oContXMean_x3_s3':          {'rtype': float, 'default': 0.},
        'oContXMean_x4_s3':          {'rtype': float, 'default': 0.},
        'oContXMean_x5_s3':          {'rtype': float, 'default': 0.},
        'oContYMean_x1_s1':          {'rtype': float, 'default': 0.},
        'oContYMean_x2_s1':          {'rtype': float, 'default': 0.},
        'oContYMean_x3_s1':          {'rtype': float, 'default': 0.},
        'oContYMean_x4_s1':          {'rtype': float, 'default': 0.},
        'oContYMean_x5_s1':          {'rtype': float, 'default': 0.},
        'oContYMean_x1_s2':          {'rtype': float, 'default': 0.},
        'oContYMean_x2_s2':          {'rtype': float, 'default': 0.},
        'oContYMean_x3_s2':          {'rtype': float, 'default': 0.},
        'oContYMean_x4_s2':          {'rtype': float, 'default': 0.},
        'oContYMean_x5_s2':          {'rtype': float, 'default': 0.},
        'oContYMean_x1_s3':          {'rtype': float, 'default': 0.},
        'oContYMean_x2_s3':          {'rtype': float, 'default': 0.},
        'oContYMean_x3_s3':          {'rtype': float, 'default': 0.},
        'oContYMean_x4_s3':          {'rtype': float, 'default': 0.},
        'oContYMean_x5_s3':          {'rtype': float, 'default': 0.},
        'oContLayerMean_x1_s1':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x2_s1':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x3_s1':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x4_s1':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x5_s1':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x1_s2':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x2_s2':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x3_s2':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x4_s2':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x5_s2':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x1_s3':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x2_s3':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x3_s3':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x4_s3':      {'rtype': float, 'default': 0.},
        'oContLayerMean_x5_s3':      {'rtype': float, 'default': 0.},
        'oContXStd_x1_s1':           {'rtype': float, 'default': 0.},
        'oContXStd_x2_s1':           {'rtype': float, 'default': 0.},
        'oContXStd_x3_s1':           {'rtype': float, 'default': 0.},
        'oContXStd_x4_s1':           {'rtype': float, 'default': 0.},
        'oContXStd_x5_s1':           {'rtype': float, 'default': 0.},
        'oContXStd_x1_s2':           {'rtype': float, 'default': 0.},
        'oContXStd_x2_s2':           {'rtype': float, 'default': 0.},
        'oContXStd_x3_s2':           {'rtype': float, 'default': 0.},
        'oContXStd_x4_s2':           {'rtype': float, 'default': 0.},
        'oContXStd_x5_s2':           {'rtype': float, 'default': 0.},
        'oContXStd_x1_s3':           {'rtype': float, 'default': 0.},
        'oContXStd_x2_s3':           {'rtype': float, 'default': 0.},
        'oContXStd_x3_s3':           {'rtype': float, 'default': 0.},
        'oContXStd_x4_s3':           {'rtype': float, 'default': 0.},
        'oContXStd_x5_s3':           {'rtype': float, 'default': 0.},
        'oContYStd_x1_s1':           {'rtype': float, 'default': 0.},
        'oContYStd_x2_s1':           {'rtype': float, 'default': 0.},
        'oContYStd_x3_s1':           {'rtype': float, 'default': 0.},
        'oContYStd_x4_s1':           {'rtype': float, 'default': 0.},
        'oContYStd_x5_s1':           {'rtype': float, 'default': 0.},
        'oContYStd_x1_s2':           {'rtype': float, 'default': 0.},
        'oContYStd_x2_s2':           {'rtype': float, 'default': 0.},
        'oContYStd_x3_s2':           {'rtype': float, 'default': 0.},
        'oContYStd_x4_s2':           {'rtype': float, 'default': 0.},
        'oContYStd_x5_s2':           {'rtype': float, 'default': 0.},
        'oContYStd_x1_s3':           {'rtype': float, 'default': 0.},
        'oContYStd_x2_s3':           {'rtype': float, 'default': 0.},
        'oContYStd_x3_s3':           {'rtype': float, 'default': 0.},
        'oContYStd_x4_s3':           {'rtype': float, 'default': 0.},
        'oContYStd_x5_s3':           {'rtype': float, 'default': 0.},
        'oContLayerStd_x1_s1':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x2_s1':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x3_s1':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x4_s1':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x5_s1':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x1_s2':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x2_s2':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x3_s2':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x4_s2':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x5_s2':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x1_s3':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x2_s3':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x3_s3':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x4_s3':       {'rtype': float, 'default': 0.},
        'oContLayerStd_x5_s3':       {'rtype': float, 'default': 0.},
        # MIP tracking variables
        'nStraightTracks':           {'rtype': int, 'default': 0},
        #'nLinregTracks':             {'rtype': int, 'default': 0},
        'firstNearPhLayer':          {'rtype': int, 'default': 33},
        'epAng':                     {'rtype': float, 'default': 0.},
        'epSep':                     {'rtype': float, 'default': 0.}
        }

def main():

    # Inputs and their trees and stuff
    pdict = manager.parse()
    inlist = pdict['inlist']
    outlist = pdict['outlist']
    group_labels = pdict['groupls']
    maxEvent = pdict['maxEvent']

    # Construct tree processes
    procs = []
    for gl, group in zip(group_labels, inlist):
        procs.append( manager.TreeProcess(event_process, group, ID=gl, pfreq=100) )

    # Process jobs
    for proc in procs:

        # Move into appropriate scratch dir
        os.chdir(proc.tmp_dir)

        # Branches needed
        proc.ecalVeto     = proc.addBranch('EcalVetoResult', 'EcalVeto_v12')
        proc.targetSPHits = proc.addBranch('SimTrackerHit', 'TargetScoringPlaneHits_v12')
        proc.ecalSPHits   = proc.addBranch('SimTrackerHit', 'EcalScoringPlaneHits_v12')
        proc.ecalRecHits  = proc.addBranch('EcalHit', 'EcalRecHits_v12')

        # Tree/Files(s) to make
        print('\nRunning %s'%(proc.ID))
        proc.tfMaker = manager.TreeMaker(group_labels[procs.index(proc)]+'.root',\
                                         "EcalVeto",\
                                         branches_info,\
                                         outlist[procs.index(proc)]
                                         )

        # RUN
        proc.extraf = proc.tfMaker.wq # Gets executed at the end of run()
        proc.run(maxEvents=maxEvent)

    # Remove scratch directory if there is one
    manager.rmScratch()

    print('\nDone!\n')


# Process an event
def event_process(self):

    # Initialize BDT input variables w/ defaults
    feats = self.tfMaker.resetFeats()

    # Assign pre-computed variables
    feats['nReadoutHits']       = self.ecalVeto.getNReadoutHits()
    feats['summedDet']          = self.ecalVeto.getSummedDet()
    feats['summedTightIso']     = self.ecalVeto.getSummedTightIso()
    feats['maxCellDep']         = self.ecalVeto.getMaxCellDep()
    feats['showerRMS']          = self.ecalVeto.getShowerRMS()
    feats['xStd']               = self.ecalVeto.getXStd()
    feats['yStd']               = self.ecalVeto.getYStd()
    feats['avgLayerHit']        = self.ecalVeto.getAvgLayerHit()
    feats['stdLayerHit']        = self.ecalVeto.getStdLayerHit()
    feats['deepestLayerHit']    = self.ecalVeto.getDeepestLayerHit() 
    feats['ecalBackEnergy']     = self.ecalVeto.getEcalBackEnergy()
    
    ###################################
    # Compute extra BDT input variables
    ###################################

    # Get e position and momentum, and make note of presence
    e_ecalHit = physTools.electronEcalSPHit(self.ecalSPHits)
    if e_ecalHit != None:
        e_present = True
        e_ecalPos, e_ecalP = e_ecalHit.getPosition(), e_ecalHit.getMomentum()
    else:
        e_present = False

    # Get electron and photon trajectories
    e_traj = g_traj = None
    if e_present:

        # Photon Info from target
        e_targetHit = physTools.electronTargetSPHit(self.targetSPHits)
        if e_targetHit != None:
            g_targPos, g_targP = physTools.gammaTargetInfo(e_targetHit)
        else:  # Should about never happen -> division by 0 in g_traj
            print('no e at targ!')
            g_targPos = g_targP = np.zeros(3)
        
        e_traj = physTools.layerIntercepts(e_ecalPos, e_ecalP)
        g_traj = physTools.layerIntercepts(g_targPos, g_targP)

    # Recoil electron momentum magnitude and angle with z-axis
    recoilPMag = physTools.mag(  e_ecalP ) if e_present      else -1.0
    recoilTheta =    physTools.angle(e_ecalP, units='radians') if recoilPMag > 0 else -1.0

    # Set electron RoC binnings
    e_radii = physTools.radius68_thetalt10_plt500
    if recoilTheta < 10 and recoilPMag >= 500: e_radii = physTools.radius68_thetalt10_pgt500
    elif recoilTheta >= 10 and recoilTheta < 20: e_radii = physTools.radius68_theta10to20
    elif recoilTheta >= 20: e_radii = physTools.radius68_thetagt20

    # Always use default binning for photon RoC
    g_radii = physTools.radius68_thetalt10_plt500

    # Big data
    trackingHitList = []

    # Major ECal loop
    for hit in self.ecalRecHits:
        
        if hit.getEnergy() > 0:

            layer = physTools.layerIDofHit(hit)
            xy_pair = ( hit.getXPos(), hit.getYPos() )

            # Distance to electron trajectory
            if e_traj != None:
                xy_e_traj = ( e_traj[layer][0], e_traj[layer][1] )
                distance_e_traj = physTools.dist(xy_pair, xy_e_traj)
            else: distance_e_traj = -1.0

            # Distance to photon trajectory
            if g_traj != None:
                xy_g_traj = ( g_traj[layer][0], g_traj[layer][1] )
                distance_g_traj = physTools.dist(xy_pair, xy_g_traj)
            else: distance_g_traj = -1.0

            # Decide which longitudinal segment the hit is in and add to sums
            for i in range(0, physTools.nSegments):
                if (physTools.segLayers[i] <= layer) and (layer <= physTools.segLayers[i + 1] - 1):
                    feats['energy_s' + str(i + 1)] += hit.getEnergy()
                    feats['nHits_s' + str(i + 1)] += 1
                    feats['xMean_s' + str(i + 1)] += xy_pair[0]*hit.getEnergy()
                    feats['yMean_s' + str(i + 1)] += xy_pair[1]*hit.getEnergy()
                    feats['layerMean_s' + str(i + 1)] += layer*hit.getEnergy()

            # Decide which containment region/segment the hit is in and add to sums
            for i in range(0, physTools.nRegions):

                if (i*e_radii[layer] <= distance_e_traj) and (distance_e_traj < (i + 1)*e_radii[layer]):
                    for j in range(0, physTools.nSegments):
                        if (physTools.segLayers[j] <= layer) and (layer <= physTools.segLayers[j + 1] - 1):
                            feats['eContEnergy_x' + str(i + 1) + '_s' + str(j + 1)] += hit.getEnergy()
                            feats['eContNHits_x' + str(i + 1) + '_s' + str(j + 1)] += 1
                            feats['eContXMean_x' + str(i + 1) + '_s' + str(j + 1)] += xy_pair[0]*hit.getEnergy()
                            feats['eContYMean_x' + str(i + 1) + '_s' + str(j + 1)] += xy_pair[1]*hit.getEnergy()
                            feats['eContLayerMean_x' + str(i + 1) + '_s' + str(j + 1)] += layer*hit.getEnergy()

                if (i*g_radii[layer] <= distance_g_traj) and (distance_g_traj < (i + 1)*g_radii[layer]):
                    for j in range(0, physTools.nSegments):
                        if (physTools.segLayers[j] <= layer) and (layer <= physTools.segLayers[j + 1] - 1):
                            feats['gContEnergy_x' + str(i + 1) + '_s' + str(j + 1)] += hit.getEnergy()
                            feats['gContNHits_x' + str(i + 1) + '_s' + str(j + 1)] += 1
                            feats['gContXMean_x' + str(i + 1) + '_s' + str(j + 1)] += xy_pair[0]*hit.getEnergy()
                            feats['gContYMean_x' + str(i + 1) + '_s' + str(j + 1)] += xy_pair[1]*hit.getEnergy()
                            feats['gContLayerMean_x' + str(i + 1) + '_s' + str(j + 1)] += layer*hit.getEnergy()

                if (distance_e_traj > (i + 1)*e_radii[layer]) and (distance_g_traj > (i + 1)*g_radii[layer]):
                    for j in range(0, physTools.nSegments):
                        if (physTools.segLayers[j] <= layer) and (layer <= physTools.segLayers[j + 1] - 1):
                            feats['oContEnergy_x' + str(i + 1) + '_s' + str(j + 1)] += hit.getEnergy()
                            feats['oContNHits_x' + str(i + 1) + '_s' + str(j + 1)] += 1
                            feats['oContXMean_x' + str(i + 1) + '_s' + str(j + 1)] += xy_pair[0]*hit.getEnergy()
                            feats['oContYMean_x' + str(i + 1) + '_s' + str(j + 1)] += xy_pair[1]*hit.getEnergy()
                            feats['oContLayerMean_x' + str(i + 1) + '_s' + str(j + 1)] += layer*hit.getEnergy()

            # Build MIP tracking hit list; (outside electron region or electron missing)
            if distance_e_traj >= e_radii[layer] or distance_e_traj == -1.0:
                hitData = physTools.HitData()
                hitData.pos = np.array( [xy_pair[0],xy_pair[1],hit.getZPos() ] )
                hitData.layer = layer
                trackingHitList.append(hitData) 

    # If possible, quotient out the total energy from the means
    for i in range(0, physTools.nSegments):
        if feats['energy_s' + str(i + 1)] > 0:
            feats['xMean_s' + str(i + 1)] /= feats['energy_s' + str(i + 1)]
            feats['yMean_s' + str(i + 1)] /= feats['energy_s' + str(i + 1)]
            feats['layerMean_s' + str(i + 1)] /= feats['energy_s' + str(i + 1)]

    for i in range(0, physTools.nRegions):
        for j in range(0, physTools.nSegments):

            if feats['eContEnergy_x' + str(i + 1) + '_s' + str(j + 1)] > 0:
                feats['eContXMean_x' + str(i + 1) + '_s' + str(j + 1)] /= feats['eContEnergy_x' + str(i + 1) + '_s' + str(j + 1)]
                feats['eContYMean_x' + str(i + 1) + '_s' + str(j + 1)] /= feats['eContEnergy_x' + str(i + 1) + '_s' + str(j + 1)]
                feats['eContLayerMean_x' + str(i + 1) + '_s' + str(j + 1)] /= feats['eContEnergy_x' + str(i + 1) + '_s' + str(j + 1)]

            if feats['gContEnergy_x' + str(i + 1) + '_s' + str(j + 1)] > 0:
                feats['gContXMean_x' + str(i + 1) + '_s' + str(j + 1)] /= feats['gContEnergy_x' + str(i + 1) + '_s' + str(j + 1)]
                feats['gContYMean_x' + str(i + 1) + '_s' + str(j + 1)] /= feats['gContEnergy_x' + str(i + 1) + '_s' + str(j + 1)]
                feats['gContLayerMean_x' + str(i + 1) + '_s' + str(j + 1)] /= feats['gContEnergy_x' + str(i + 1) + '_s' + str(j + 1)]

            if feats['oContEnergy_x' + str(i + 1) + '_s' + str(j + 1)] > 0:
                feats['oContXMean_x' + str(i + 1) + '_s' + str(j + 1)] /= feats['oContEnergy_x' + str(i + 1) + '_s' + str(j + 1)]
                feats['oContYMean_x' + str(i + 1) + '_s' + str(j + 1)] /= feats['oContEnergy_x' + str(i + 1) + '_s' + str(j + 1)]
                feats['oContLayerMean_x' + str(i + 1) + '_s' + str(j + 1)] /= feats['oContEnergy_x' + str(i + 1) + '_s' + str(j + 1)]
    
    # MIP tracking starts here

    # Goal: Calculate 
    # nStraightTracks (Self-explanatory) 
    # nLinregTracks (Tracks found by linreg algorithm)

    # Find epAng and epSep, and prepare EP trajectory vectors
    if e_traj != None and g_traj != None:

        # Create arrays marking start and end of each trajectory
        e_traj_ends = [np.array([e_traj[0][0], e_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([e_traj[-1][0], e_traj[-1][1], physTools.ecal_layerZs[-1] ])]
        g_traj_ends = [np.array([g_traj[0][0], g_traj[0][1], physTools.ecal_layerZs[0]    ]),
                       np.array([g_traj[-1][0], g_traj[-1][1], physTools.ecal_layerZs[-1] ])]

        evec   = e_traj_ends[1] - e_traj_ends[0]
        gvec   = g_traj_ends[1] - g_traj_ends[0]
        e_norm = physTools.unit(evec)
        g_norm = physTools.unit(gvec)

        # Unused epAng and epSep ??? And why Ang instead of dot ???
        feats['epAng'] = epAng = math.acos( physTools.dot(e_norm,g_norm) )*180.0/math.pi
        feats['epSep'] = epSep = physTools.dist( e_traj_ends[0], g_traj_ends[0] )

    else:

        # Electron trajectory is missing so all hits in Ecal are okay to use
        # Pick trajectories so they won'trestrict tracking, far outside the Ecal

        e_traj_ends   = [np.array([999 ,999 ,0   ]), np.array([999 ,999 ,999 ]) ]
        g_traj_ends   = [np.array([1000,1000,0   ]), np.array([1000,1000,1000]) ]
        feats['epAng'] = epAng = 3.0 + 1.0 # Don't cut on these in this case
        feats['epSep'] = epSep = 10.0 + 1.0

    # Near photon step: Find the first layer of the ECal where a hit near the projected
    # photon trajectory is found
    # Currently unusued pending further study; performance has dropped between v9 and v12
    if g_traj != None: # If no photon trajectory, leave this at the default
        feats['firstNearPhLayer'] = mipTracking.firstNearPhLayer( trackingHitList, g_traj )

    # Order hits by zpos for efficiency
    trackingHitList.sort(key=lambda hd: hd.layer, reverse=True)

    # Find MIP tracks
    feats['nStraightTracks'], trackingHitList = mipTracking.nStraightTracks(trackingHitList,
                                                        e_traj_ends, e_traj_ends)
    #feats['nLinregTracks'] = mipTracking.nLinregTracks( trackingHitList,
    #                                                    e_traj_ends, e_traj_ends)

    # Fill the tree with values for this event
    self.tfMaker.fillEvent(feats)

if __name__ == "__main__":
    main()
