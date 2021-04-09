"""Histogramming"""

import ROOT
import bdt
import sys
import argparse

class HistogramPool() :
    def __init__(self) :
        self.__dict__["category"] = 'nuc'
        self.weight   = 1.

    def __full_name__(self, name) :
        if name == 'category' or name == 'weight' :
            return name
        else :
            return f'{name}_{self.__dict__["category"]}'

    def __getattr__(self, hist_name) :
        return self.__dict__[self.__full_name__(hist_name)]

    def __setattr__(self, hist_name, hist) :
        super().__setattr__(self.__full_name__(hist_name), hist)

# Parse
parser = argparse.ArgumentParser(f'ldmx python3 {sys.argv[0]}', 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('input_file',nargs='+',help='File(s) to run BDT Evaluator on')
parser.add_argument('--out',required=True,help='Name to write output histgorams to.')
parser.add_argument('--bdt',default=None,help='BDT to use in evaluation')
parser.add_argument('--bdt_cut',default=0.9,type=float,help='Analysis cut on BDT Value')
parser.add_argument('--pe_cut',default=15,type=float,help='Analysis cut on HCal Max PE')

arg = parser.parse_args()

p = HistogramPool()

output_file = ROOT.TFile.Open(arg.out,'RECREATE')
eatAna_d = output_file.mkdir('eatAna')
for c in ['nuc','ap1','ap5','ap10','ap50','ap100','ap500','ap1000'] :
    d = eatAna_d.mkdir(c)
    d.cd()

    p.category = c

    p.ecal_bdt = ROOT.TH1F('ecal_bdt',';ECal BDT Disc',100,0.,1.)
    p.ecal_bdt__hcal_pe = ROOT.TH2F('ecal_bdt__hcal_pe',';ECal BDT Disc;Max HCal PE',
        100,0.,1.,300,0.,300.)
    p.hcal_side__back_fail_layer_bdt = ROOT.TH2F("hcal_side__back_fail_layer_bdt",
          ";Min Side Layer above PE Cut;Min Back Layer above PE Cut",
          31,-1,30,101,-1,100);
    p.ecal_n_straight_tracks = ROOT.TH1F('ecal_n_straight_tracks',';ECal N Straight Tracks',10,-0.5,9.5)
    p.ecal_n_linreg_tracks = ROOT.TH1F('ecal_n_linreg_tracks',';ECal N Lin Reg Tracks',10,-0.5,9.5)

    p.ecal_bdt__tracks = ROOT.TH2F('ecal_bdt__tracks',';ECal BDT Disc;N Tracks',
        100,0.,1.,10,-0.5,9.5)

    p.ecal_tracks__hcal_pe = ROOT.TH2F('ecal_tracks__hcal_pe',';N ECal Tracks;Max HCal PE',
        10,-0.5,9.5,50,0,50)
    
# create histograms

evaluator = None
if arg.bdt is not None :
  evaluator = bdt.Evaluator(arg.bdt)

input_sample = bdt.SampleContainer(bdt.smart_recursive_input(arg.input_file))

sim_particles = ROOT.std.map(int,'ldmx::SimParticle')()
input_sample.__attach__(sim_particles, 'SimParticles')

hcal_hits = ROOT.std.vector('ldmx::HcalHit')()
input_sample.__attach__(hcal_hits, 'HcalRecHits')

for e in input_sample.t :
    # Determine weight and category
    w = e.EventHeader.getWeight()

    p.category = 'nuc'
    for part in sim_particles :
        if part.second.getPdgID() == 622 :
            p.category = f'ap{int(part.second.getMass())}'
            break
        #found A'
    #search particles for A'

    # Get Maximum Hcal PE
    max_back_pe = 0
    max_side_pe = 0
    min_layer_back = -1
    min_layer_side = -1
    for hit in hcal_hits :
        pe = hit.getPE()

        # manually decode ID
        i  = hit.getID()
        section = (i >> 18) & 0x7
        layer   = (i >> 10) & 0xFF
        
        if section == 0 :
            if pe > max_back_pe : max_back_pe = pe
            if pe > arg.pe_cut and (layer < min_layer_back or min_layer_back < 0) :
                min_layer_back = layer
        else :
            if pe > max_side_pe : max_side_pe = pe
            if pe > arg.pe_cut and (layer < min_layer_side or min_layer_side < 0) :
                min_layer_side = layer

    # Calculate BDT disc value
    #   (or get it from the file)
    if evaluator is None :
        d = input_sample.veto.getDisc()
    else :
        d = evaluator.__eval__(input_sample.single_translation())

    p.ecal_n_straight_tracks.Fill(input_sample.veto.getNStraightTracks(), w)
    p.ecal_n_linreg_tracks.Fill(input_sample.veto.getNLinRegTracks(), w)
    p.ecal_bdt.Fill(d, w)

    max_hcal_pe = max(max_back_pe,max_side_pe)
    track_count = input_sample.veto.getNStraightTracks()+input_sample.veto.getNLinRegTracks()

    p.ecal_bdt__hcal_pe.Fill(d, max_hcal_pe, w)
    p.ecal_bdt__tracks.Fill(d,track_count,w)
    p.ecal_tracks__hcal_pe.Fill(track_count,max_hcal_pe,w)
    if d > arg.bdt_cut and track_count == 0:
        p.hcal_side__back_fail_layer_bdt.Fill(min_layer_side, min_layer_back, w)
#loop through events

output_file.Write()
output_file.Close()
