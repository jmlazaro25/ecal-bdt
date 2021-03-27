import os
import sys
import ROOT as r
import numpy as np # ?


#TODO: Add a move to scratch dir function
#TODO: Make options for no output or based on input
#TODO: Make nolists independant for in and out

###################################
# Constants
###################################

# ROOT colors
colors = {
        'kBlue': 600,
        'kGreen': 417,
        'kMagenta': 616,
        'kOrange': 807, # +7
        'kBlack': 1,
        'kYellow': 400,
        'kViolet': 880,
        'kRed': 632,
        'kCyan': 432
        }

# For easier loops
color_list = [colors[key] for key in colors]

# ROOT 1D/2D line styles
lineStyles = {
        'kSolid': 1,
        'kDashed': 2,
        'kDotted': 3,
        'kDashDotted': 4
        }

# For easier loops
lineStyle_list = [i for i in range(1,11)]


###################################
# Classes
###################################

class TreeProcess:

    # For analysing .root samples

    def __init__(self, event_process, group=[], tree=None, tree_name = None, ID = '',\
            color=1, maxEvents=-1, pfreq=1000, interactive=True, extraf=None):

        self.event_process = event_process
        self.group_files = group
        self.tree = tree
        self.tree_name = tree_name
        self.ID = ID
        self.color = color
        self.maxEvents = maxEvents
        self.pfreq = pfreq
        self.interactive = interactive
        self.extraf = extraf
        self.cwd = os.getcwd()

        
        # Build tree amd move operations to a scratch directory
        # if providing group_files instead of a tree

        self.mvd = False
        if self.tree == None:
            self.mvd = True

            # Create the scratc directory if it doesn't already exist
            scratch_dir='./scratch'
            print( 'Using scratch path %s' % scratch_dir )
            if not os.path.exists(scratch_dir):
                os.makedirs(scratch_dir)

            # Create and mv into tmp directory that can be used to copy files into
            if self.interactive:
                self.tmp_dir = '%s/%s' % (scratch_dir, 'tmp')
            else:
                self.tmp_dir='%s/%s' % (scratch_dir, os.environ['LSB_JOBID'])
            if not os.path.exists(self.tmp_dir):
                print( 'Creating tmp directory %s' % self.tmp_dir )
            os.makedirs(self.tmp_dir)
            os.chdir(self.tmp_dir)
    
            # Copy input files to the tmp directory
            print( 'Copying input files into tmp directory' )
            for rfilename in self.group_files:
                os.system("cp %s ." % rfilename )
            os.system("ls .")
    
            # Just get the file names without the full path
            tmpfiles = [f.split('/')[-1] for f in self.group_files]
    
            # Load'em
            if self.tree_name != None:
                self.tree = load(tmpfiles, self.tree_name)
            else:
                self.tree = load(tmpfiles)

    def addBranch(self, ldmx_class, branch_name):

        # Add a new branch to read from

        if self.tree == None:
            sys.exit('Set tree')

        if ldmx_class == 'EventHeader':
            branch = r.ldmx.EventHeader()
        elif ldmx_class == 'SimParticle':
            branch = r.map(int, 'ldmx::'+ldmx_class)() 
        else:
            branch = r.std.vector('ldmx::'+ldmx_class)()

        self.tree.SetBranchAddress(branch_name,r.AddressOf(branch))

        return branch
 
    def run(self, maxEvents=-1, pfreq=1000):
   
        # Process events

        if maxEvents != -1: self.maxEvents = maxEvents
        else: self.maxEvents = self.tree.GetEntries()
        if pfreq != 1000: self.pfreq = pfreq
        
        self.event_count = 0
        while self.event_count < self.maxEvents:
            self.tree.GetEntry(self.event_count)
            if self.event_count%self.pfreq == 0:
                print('Processing Event: %s'%(self.event_count))
            self.event_process(self)
            self.event_count += 1

        # Execute any closing function(s)
        if self.extraf != None:
            self.extraf()

        # Move back to cwd in case running multiple procs
        os.chdir(self.cwd)

        # Remove tmp directory if created in move
        if self.mvd:
            print( 'Removing tmp directory %s' % self.tmp_dir )
            os.system('rm -rf %s' % self.tmp_dir)

class TreeMaker:

    # To write a tree in an analysis process

    def __init__(self, outfile, tree_name, branches_info = {}, outdir=''):

        self.outfile = outfile
        self.tree_name = tree_name
        self.branches_info = branches_info
        self.branches = {}
        self.outdir = outdir

        # Create output file and tree
        self.tfout = r.TFile(self.outfile,"RECREATE")
        self.tree = r.TTree(tree_name, tree_name)

        # Set up new tree branches if given branches_info
        if len(branches_info) != 0:
            for branch_name in branches_info:
                self.addBranch(self.branches_info[branch_name]['rtype'],\
                               self.branches_info[branch_name]['default'],\
                               branch_name)

    def addBranch(self, rtype, default_value, branch_name):

        # Add a new branch to write to

        self.branches_info[branch_name] = {'rtype': rtype, 'default': default_value}
        self.branches[branch_name] = np.zeros(1, dtype=rtype)
        if str(rtype) == "<type 'float'>" or str(rtype) == "<class 'float'>":
            self.tree.Branch(branch_name, self.branches[branch_name], branch_name + "/D")
        elif str(rtype) == "<type 'int'>" or str(rtype) == "<class 'int'>":
            self.tree.Branch(branch_name, self.branches[branch_name], branch_name + "/I")
        # ^ probably use cases based on rtype to change the /D if needed?

    def resetBranches(self):

        # Reset variables to defaults for new event
        # Return because feats['feat'] looks nicer than self.tfMaker.feats['feat']

        feats = {}
        for branch_name in self.tfMaker.branches_info:
            feats[branch_name] = self.tfMaker.branches_info[branch_name]['default']

        return feats

    def fillEvent(self, feats):

        # Fill the tree with new feature values

        for feat in feats:
            self.branches[feat][0] = feats[feat]

        self.tree.Fill()

    def wq(self):

        # Save the tree and close the file

        self.tree.Write()
        self.tfout.Close()

        if self.outdir != '':
            if not os.path.exists(self.outdir):
                print( 'Creating %s' % (self.outdir) )
                os.makedirs(self.outdir)
            print( 'cp %s %s' % (self.outfile,self.outdir) )
            os.system('cp %s %s' % (self.outfile,self.outdir))

class Histogram:

    # Just to hold histogram-related stuff and make other py code nicer

    def __init__(self, hist, title='', xlabel='x', ylabel='y',\
            color=1, lineStyle=1, fillStyle=1):
        self.hist = hist
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.color = color
        self.lineStyle = lineStyle
        self.fillStyle = fillStyle


###################################
# Functions
###################################

def parse(nolist = False):

    import glob
    import argparse

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true',
            help='Run in interactive mode [Default: False]')
    parser.add_argument('-i', nargs='+', action='store', dest='infiles', default=[],
            help='input file(s)')
    parser.add_argument('--indirs', nargs='+', action='store', dest='indirs', default=[],
            help='Director(y/ies) of input files')
    parser.add_argument('-g','-groupls', nargs='+', action='store', dest='group_labels',
            default='', help='Human readable sample labels e.g. for legends')
    parser.add_argument('--out', nargs='+', action='store', dest='out', default=[],
            help='output files or director(y/ies) of output files')
            # if inputting directories, it's best to make a system
            # for naming files in main and just provide 
    parser.add_argument('--notlist', action='store_true', dest='nolist',
            help="return things without lists (to make things neater for 1 sample runs")
    parser.add_argument('-m','--max', type=int, action='store', dest='maxEvent',
            default=-1, help='max events to run over for EACH group')
    args = parser.parse_args()

    # Input
    if args.infiles != []:
        inlist = [[f] for f in args.infiles] # Makes general loading easier
        if nolist or args.nolist == True:
            inlist = inlist[0]
    elif args.indirs != []:
        inlist = [glob.glob(indir + '/*.root') for indir in args.indirs]
        if nolist or args.nolist == True:
            inlist = inlist[0]
    else:
        sys.exit('provide input')

    # Output
    if args.out != []:
        outlist = args.out
        if nolist or args.nolist == True:
            outlist = outlist[0]
    else:
        sys.exit('provide output')
    
    pdict = {
            'inlist':inlist,
            'groupls':args.group_labels,
            'outlist':outlist,
            'maxEvent':args.maxEvent
            }

    return pdict


def load(group,treeName='LDMX_Events'):

    # Load a group of files into a readable tree

    tree = r.TChain(treeName)
    for f in group:
        tree.Add(f)

    return tree


