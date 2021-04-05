
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

full_input_list = []

def recursive_input(file_or_dir) :
    """Recursively add the full path to the file or files in the input directory"""
    if os.path.isfile(file_or_dir) and file_or_dir.endswith('.root') :
        full_input_list.append(os.path.realpath(file_or_dir))
    elif os.path.isdir(file_or_dir) :
        for f in os.listdir(file_or_dir) :
            recursive_input(os.path.join(file_or_dir,f))
        #end loop over files in dir
    else :
        print("'%s' is not a ROOT file or a directory. Skipping."%(file_or_dir))
    #file or directory

for input_file in arg.input_file :
    recursive_input(input_file)

import bdt

print('Loading BDT into memory...')
e = bdt.Evaluator(arg.bdt)

for sub_list in [ full_input_list[i:i+arg.files_per_job] for i in range(0,len(full_input_list),arg.files_per_job) ] :
    print('Loading sample to evaluate...')
    sample = bdt.SampleContainer(sub_list)
    
    print('Evaluating BDT...')
    e.eval(sample, max_events=arg.max_evt, out_name=f'{arg.out_dir}/bdt_{len(sub_list)}_{os.path.basename(sub_list[0])}')
#done evaluation 

print('Done')
