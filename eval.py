
import os
import sys
import argparse

# Parse
parser = argparse.ArgumentParser(f'ldmx python3 {sys.argv[0]}', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('input_file',nargs='+',help='File to run BDT Evaluator on')
parser.add_argument('--bdt',required=True,help='BDT to use in evaluation')

parser.add_argument('--max_evt',type=int,default=None,help='Max Events to load')
parser.add_argument('--out_dir',default='.',help='Directory to write output events to.')
parser.add_argument('--files_per_job',default=1,type=int,help='Number files to merge together during evaluation.')

arg = parser.parse_args()

import bdt

print('Loading BDT into memory...')
e = bdt.Evaluator(arg.bdt)

print('Getting file listing...')
full_input_list = list(bdt.smart_recursive_input(arg.input_file))

for sub_list in [ full_input_list[i:i+arg.files_per_job] for i in range(0,len(full_input_list),arg.files_per_job) ] :
    print('Loading sample to evaluate...')
    sample = bdt.SampleContainer(sub_list)
    
    print('Evaluating BDT...')
    e.eval(sample, max_events=arg.max_evt, out_name=f'{arg.out_dir}/bdt_{len(sub_list)}_{os.path.basename(sub_list[0])}')
#done evaluation 

print('Done')
