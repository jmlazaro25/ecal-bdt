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

arg = parser.parse_args()

p = HistogramPool()

output_file = ROOT.TFile.Open(f'hists_{arg.out}.root','RECREATE')
eatAna_d = output_file.mkdir('eatAna')
for c in ['nuc','ap1','ap5','ap10','ap50','ap100','ap500','ap1000'] :
    d = eatAna_d.mkdir(c)
    d.cd()

    p.category = c

    p.ecal_bdt = ROOT.TH1F('ecal_bdt',';ECal BDT Disc',100,0.,1.)
    p.ecal_bdt__hcal_pe = ROOT.TH2F('ecal_bdt__hcal_pe',';ECal BDT Disc;Max HCal PE',
        100,0.,1.,300,0.,300.)
    
# create histograms

evaluator = None
if arg.bdt is not None :
  evaluator = bdt.Evaluator(arg.bdt)
input_sample = bdt.SampleContainer(arg.input_file)

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
    pe = 0
    for hit in hcal_hits :
        if hit.getPE() > pe :
            pe = hit.getPE()

    # Calculate BDT disc value
    #   (or get it from the file)
    if evaluator is None :
        d = input_sample.veto.getDisc()
    else :
        d = evaluator.__eval__(input_sample.single_translation())

    p.ecal_bdt.Fill(d, w)
    p.ecal_bdt__hcal_pe.Fill(d, pe, w)
#loop through events

for c in ['nuc','ap1','ap5','ap10','ap50','ap100','ap500','ap1000'] :
    p.category = c

    p.ecal_bdt.Write()
    p.ecal_bdt__hcal_pe.Write()
#write histograms
output_file.Write()
output_file.Close()
