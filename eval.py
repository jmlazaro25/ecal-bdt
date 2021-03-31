
import os
import sys
import argparse

# Parse
parser = argparse.ArgumentParser(f'ldmx python3 {sys.argv[0]}', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('input_file',help='File to run BDT Evaluator on')
parser.add_argument('--bdt',required=True,help='BDT to use in evaluation')

parser.add_argument('--max_evt', dest='max_evt',type=int,default=None,help='Max Events to load')

arg = parser.parse_args()

import bdt

# Make Signal Container
print( f'Loading {arg.input_file}' )
sample = bdt.SampleContainer(arg.input_file,0,True)

e = bdt.Evaluator(arg.bdt)

e.eval(sample, max_events=arg.max_evt, out_name=f'bdt_{os.path.basename(arg.input_file)}')

print('Done')
