
import sys
import argparse

# Parse
parser = argparse.ArgumentParser(f'ldmx python3 {sys.argv[0]}', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-b',metavar='BKGD_FILE',dest='bkg_file',required=True,help='Path to background file')
parser.add_argument('-s',metavar='SIGN_FILE',dest='sig_file',required=True,help='Path to signal file')
parser.add_argument('-o',metavar='OUT_NAME',dest='out_name',required=True,help='Output name of trained BDT')

parser.add_argument('--seed', dest='seed',type=int,default=2,help='Numpy random seed.')
parser.add_argument('--max_evt', dest='max_evt',type=int,default=1500000, help='Max Events to load')
parser.add_argument('--eta', dest='eta',type=float,default=0.023, help='Learning Rate')
parser.add_argument('--tree_number', dest='tree_number',type=int,default=1000,help='Tree Number')
parser.add_argument('--depth', dest='depth',type=int,default=10,help='Max Tree Depth')

arg = parser.parse_args()

import numpy
import bdt

# Seed numpy's randomness
numpy.random.seed(arg.seed)

# Print run info
print( f'Random seed is = {arg.seed}' )
print( f'You set max_evt = {arg.max_evt}' )
print( f'You set tree number = {arg.tree_number}' )
print( f'You set max tree depth = {arg.depth}' )
print( f'You set eta = {arg.eta}' )

# Make Signal Container
print( f'Loading sig_file = {arg.sig_file}' )
sigContainer = bdt.SampleContainer(arg.sig_file,arg.max_evt,True)

# Make Background Container
print( f'Loading bkg_file = {arg.bkg_file}' )
bkgContainer = bdt.SampleContainer(arg.bkg_file,arg.max_evt,False)

t = bdt.Trainer(sigContainer, bkgContainer, arg.eta, arg.depth, arg.tree_number)

t.train()

t.save(arg.out_name)

print('Done')
