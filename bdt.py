"""
Core methods for both training and evaluating the BDT
"""

import ROOT

def load_event_dict() :
    ROOT.gSystem.Load('libFramework.so')

def translation(event) :
    """Translate input ROOT event object into the feature vector."""

    try :
        veto = event.EcalVeto_eat
    except :
        veto = event.EcalVeto_signal

    return [
        veto.getNReadoutHits(),
        veto.getSummedDet(),
        veto.getSummedTightIso(),
        veto.getMaxCellDep(),
        veto.getShowerRMS(),
        veto.getXStd(),
        veto.getYStd(),
        veto.getAvgLayerHit(),
        veto.getStdLayerHit(),
        veto.getDeepestLayerHit(),
        veto.getEcalBackEnergy(),
        veto.getNStraightTracks(),
        veto.getNLinRegTracks()
        ]
