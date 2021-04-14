"""
Core methods for both training and evaluating the BDT
"""

import ROOT

import os
import numpy
import pickle
import xgboost
import matplotlib

ROOT.gSystem.Load('libFramework.so')

def smart_recursive_input(file_or_dir) :
    """Recursively add the full path to the file or files in the input directory"""
    full_list = []
    if isinstance(file_or_dir,list) :
        for entry in file_or_dir :
            full_list.extend(smart_recursive_input(entry))
    elif os.path.isfile(file_or_dir) and file_or_dir.endswith('.root') :
        full_list.append(os.path.realpath(file_or_dir))
    elif os.path.isfile(file_or_dir) and file_or_dir.endswith('.list') :
        with open(file_or_dir) as listing :
            file_listing = listing.readlines()

        full_list.extend(smart_recursive_input([f.strip() for f in file_listing]))
    elif os.path.isdir(file_or_dir) :
        full_list.extend(smart_recursive_input([os.path.join(file_or_dir,f) for f in os.listdir(file_or_dir)]))
    else :
        print(f"'{file_or_dir}' is not a ROOT file, a directory, or a list of files. Skipping.")
    #file or directory
    return full_list

class HistogramPool() :
    """Pool to keep histograms and write them to an output file.

    Each histogram has an associated category (sub directory)
    so that multiple different categories can have the same
    histograms.
    """

    def __init__(self, out_name, root_dir) :
        self.category = 'nuc'
        self.weight   = 1.
        self.f = ROOT.TFile.Open(out_name,'RECREATE')
        self.root_d = self.f.mkdir(root_dir)

    def save(self) :
        self.f.Write()
        self.f.Close()

    def new_category(self, cat) :
        d = self.root_d.mkdir(cat)
        d.cd()
        self.category = cat

    def __full_name__(self, name) :
        if name == 'category' or name == 'weight' or name == 'f' or name == 'root_d' :
            return name
        else :
            return f'{name}_{self.__dict__["category"]}'

    def __getattr__(self, hist_name) :
        return self.__dict__[self.__full_name__(hist_name)]

    def __setattr__(self, hist_name, hist) :
        super().__setattr__(self.__full_name__(hist_name), hist)

class SampleContainer:
    """Container for reading in (or writing out) the LDMX_Events tree."""

    def __init__(self, file_paths, tree_to_clone=None):
        if tree_to_clone is None :
            self.t = ROOT.TChain('LDMX_Events')
            for f in file_paths :
                if not self.t.Add(f) :
                    raise Exception(f'Error adding file {f}.')
        else :
            if len(file_paths) > 1 :
                raise Exception('Can\'t clone tree into multiple files.')

            self.f = ROOT.TFile.Open(file_paths[0],'RECREATE')
            self.t = tree_to_clone.CloneTree(0)
        # reading or writing
    
        self.veto = ROOT.ldmx.EcalVetoResult()
        try :
            self.__attach__(self.veto,'EcalVeto','anaeat')
        except :
            self.__attach__(self.veto,'EcalVeto')

    def __attach__(self,obj,coll_name,pass_name='') :
        """Attach an object to a specific branch of the event tree.

        We look for a unique match. Otherwise an exception is thrown.

        We assume that the object being attached is the correct type.
        """
        
        full_name = f'{coll_name}_{pass_name}'

        options = [b.GetName() for b in self.t.GetListOfBranches() if full_name in b.GetName()]

        if len(options) == 0 :
            raise Exception(f'No branch matching {full_name}.')
        elif len(options) > 1 :
            raise Exception(f'More than one branch matching {full_name}.')

        # now we know we have a unique branch 'options[0]'
        self.t.SetBranchAddress(options[0], ROOT.AddressOf(obj))

    def single_translation(self) :
        """Translate the current event into a feature vector for the BDT"""
        return [
            self.veto.getSummedDet(),
            self.veto.getShowerRMS(),
            self.veto.getAvgLayerHit(),
            self.veto.getStdLayerHit(),
            self.veto.getEcalBackEnergy(),
            self.veto.getNStraightTracks(),
            self.veto.getNLinRegTracks()
            ]

    def translation(self, max_events = None, shuffle = True) :
        """Translating the sample wrapped by this container into list of feature vectors"""

        num_events = 0
        events = []
        for event in self.t :
            num_events += 1
            if max_events is not None and num_events > max_events :
                break

            events.append(self.single_translation())

        if shuffle :
            new_idx=numpy.random.permutation(numpy.arange(numpy.shape(events)[0]))
            events = numpy.array(events)
            numpy.take(events, new_idx, axis=0, out=events)

        return events

class Trainer :
    """Given a signal and a background sample, train a BDT to try to discriminate between them.

    max_events is an option for development.

    The constructor simply reads in the samples and translates them into
    a list of feature vectors for the BDT to train on. The actual training is
    done in the train method.
    """

    def __init__(self, sig_sample, bkg_sample, max_events) :

        sign_feats = sig_sample.translation(max_events = max_events, shuffle=True)
        bkgd_feats = bkg_sample.translation(max_events = max_events, shuffle=True)

        train_x = numpy.vstack((sign_feats,bkgd_feats))
        train_y = numpy.append(numpy.zeros(len(sign_feats))+1,numpy.zeros(len(bkgd_feats)))
        
        train_x[numpy.isnan(train_x)] = 0.000
        train_y[numpy.isnan(train_y)] = 0.000
        
        self.unified_training_sample = xgboost.DMatrix(train_x,train_y)

    def train(self, tree_number = 1, eta = 0.023, depth = 10) :
        """Train the BDT model from the samples that we were constructed with."""

        # Actual training
        training_parameters = {'objective': 'binary:logistic',
                  'eta': eta,
                  'max_depth': depth,
                  'min_child_weigrt': 20,
                  'silent': 1,
                  'subsample':.9,
                  'colsample_bytree': .85,
                  #'eval_metric': 'auc',
                  'eval_metric': 'error',
                  'seed': 1,
                  'nthread': 1,
                  'verbosity': 1,
                  'early_stopping_rounds' : 10}
        self.gbm = xgboost.train(training_parameters, self.unified_training_sample, tree_number)

    def save(self, name, plot_importance=True) :
        """Store the current BDT model into a pickle file for later loading into Python

        Optionally, we can plot the importance of the various features of the BDT
        """
        # Store BDT
        with open(f'{name}_weights.pkl','wb') as output :
            pickle.dump(self.gbm, output)

        if not plot_importance :
            return
    
        # Plot feature importances
        xgboost.plot_importance(self.gbm)
        matplotlib.pyplot.savefig(f'{name}_fimportance.png',
                dpi=500, bbox_inches='tight', pad_inches=0.5) # png parameters
        

class Evaluator:
    """Load in a trained BDT and evaluate samples."""
    def __init__(self, pkl_file) :
        self.model = pickle.load(open(pkl_file,'rb'))

    def __eval__(self, translated_event) :
        """Evaluate the input event i.e. feature vector. We assume that the input feature
        vector is constructed in the same way that the BDT expects.
        """

        return float(self.model.predict(xgboost.DMatrix(numpy.array([translated_event])))[0])

    def eval(self, input_sample, out_name, max_events=None) :
        """Evaluate an entire sample and output the evaluated sample into a new file
        that contains the entire event tree with the updated discriminator value.
        """

        output_sample = SampleContainer([out_name], input_sample.t)
        for event in input_sample.t :
            output_sample.veto.setDiscValue(self.__eval__(input_sample.single_translation()))
            output_sample.t.Fill()

            if max_events is not None :
                max_events -= 1
                if max_events < 0 :
                    break

        #done with loop over read-in events

        output_sample.t.Write()
        output_sample.f.Close()

def eval_action(arg) :
    print('Loading BDT into memory...')
    e = Evaluator(arg.bdt)
    print('Loading sample to evaluate...')
    sample = SampleContainer(smart_recursive_input(arg.input_sample))
    print('Evaluating BDT...')
    e.eval(sample, max_events = arg.max_events, out_name = arg.out_name)
    print('Done')

def train_action(arg) :

    # Seed numpy's randomness
    numpy.random.seed(arg.seed)
    
    # Print run info
    print(f'Random seed is = {arg.seed}' )
    print(f'You set max_evt = {arg.max_evt}' )
    print(f'You set tree number = {arg.tree_number}' )
    print(f'You set max tree depth = {arg.depth}' )
    print(f'You set eta = {arg.eta}' )
    
    # Make Signal Container
    print('Loading signal files...')
    sig_sample = bdt.SampleContainer(bdt.smart_recursive_input(arg.sig_files))
    
    # Make Background Container
    print('Loading bkgd files...')
    bkg_sample = bdt.SampleContainer(bdt.smart_recursive_input(arg.bkg_files))
    
    print('Constructing trainer and translating ROOT objects...')
    t = bdt.Trainer(sig_sample,bkg_sample,arg.max_evt)
    
    print('Training...')
    t.train(arg.tree_number,arg.eta,arg.depth)
    
    print('Saving output...')
    t.save(arg.out_name)
    
    print('Done')

def ana_action(arg) :
    p = bdt.HistogramPool(arg.out, 'eatAna')
    for c in ['nuc','ap1','ap5','ap10','ap50','ap100','ap500','ap1000'] :
        p.new_category(c)
    
        p.ecal_bdt = ROOT.TH1F('ecal_bdt',';ECal BDT Disc',100,0.,1.)
        p.ecal_bdt__hcal_pe = ROOT.TH2F('ecal_bdt__hcal_pe',';ECal BDT Disc;Max HCal PE',
            100,0.,1.,300,0.,300.)
        p.hcal_side__back_fail_layer_bdt = ROOT.TH2F("hcal_side__back_fail_layer_bdt",
              ";Min Side Layer above PE Cut;Min Back Layer above PE Cut",
              31,-1,30,101,-1,100);
    
        p.ecal_bdt__tracks = ROOT.TH2F('ecal_bdt__tracks',';ECal BDT Disc;N Tracks',
            100,0.,1.,10,-0.5,9.5)
    # create histograms
    
    evaluator = None
    if arg.bdt is not None :
      evaluator = Evaluator(arg.bdt)
    
    input_sample = SampleContainer(smart_recursive_input(arg.input_file))
    
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
    
        p.ecal_bdt.Fill(d, w)
    
        max_hcal_pe = max(max_back_pe,max_side_pe)
        track_count = input_sample.veto.getNStraightTracks()+input_sample.veto.getNLinRegTracks()
    
        p.ecal_bdt__hcal_pe.Fill(d, max_hcal_pe, w)
        p.ecal_bdt__tracks.Fill(d,track_count,w)
        if d > arg.bdt_cut and track_count == 0:
            p.hcal_side__back_fail_layer_bdt.Fill(min_layer_side, min_layer_back, w)
    #loop through events
    
    p.save()

if __name__ == '__main__' :
    import os
    import sys
    import argparse
    
    # Parse
    parser = argparse.ArgumentParser(f'ldmx python3 {sys.argv[0]}', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(help='Choose which action to perform.')

    # Training Args
    parse_train = subparsers.add_parser('train', help='Train BDT with provided signal and background files.')
    parse_train.add_argument('-b',metavar='BKGD_SAMPLE',dest='bkg_files',nargs='+',required=True,help='Path to background file(s), directory(ies), or listing(s).')
    parse_train.add_argument('-s',metavar='SIGN_SAMPLE',dest='sig_files',nargs='+',required=True,help='Path to signal file(s), directory(ies), or listing(s).')
    parse_train.add_argument('-o',metavar='OUT_NAME',dest='out_name',required=True,help='What to name trained BDT.')
    parse_train.add_argument('--seed', dest='seed',type=int,default=2,help='Numpy random seed.')
    parse_train.add_argument('--eta', dest='eta',type=float,default=0.023, help='Learning Rate')
    parse_train.add_argument('--tree_number', dest='tree_number',type=int,default=1000,help='Tree Number')
    parse_train.add_argument('--depth', dest='depth',type=int,default=10,help='Max Tree Depth')
    parse_train.add_argument('--max_evt',type=int,default=None,help='Max Events to load')
    parse_train.set_defaults(action=train_action)
    
    # Evaluation Args
    parse_eval = subparsers.add_parser('eval', help='Evaluate BDT and store in a new event file.')
    parse_eval.add_argument('input_file',nargs='+',help='File to run BDT Evaluator on')
    parse_eval.add_argument('--out_dir',default='.',help='Directory to write output events to.')
    parse_eval.add_argument('--files_per_job',default=1,type=int,help='Number files to merge together during evaluation.')
    parse_eval.set_defaults(action=eval_action)

    # Analysis Args
    parse_ana = subparsers.add_parser('ana', help='Analyze performance of BDT by evaluating it and filling histograms.')
    parse_ana.add_argument('input_file',nargs='+',help='File(s), Directory(ies), or Listing(s) to run BDT Evaluator on')
    parse_ana.add_argument('--out',required=True,help='Name to write output histgorams to.')
    parse_ana.add_argument('--bdt',default=None,help='BDT to use in evaluation')
    parse_ana.add_argument('--bdt_cut',default=0.9,type=float,help='Analysis cut on BDT Value')
    parse_ana.add_argument('--pe_cut',default=15,type=float,help='Analysis cut on HCal Max PE')
    parse_ana.set_defaults(action=ana_action)
    
    arg = parser.parse_args()

    # Each sub-command defines the 'action' member of the arg namespace
    #   so if this member is not defined, no action has been selected
    if 'action' not in arg :
        parser.error('Must choose an action to perform!')
         
    arg.action(arg)

