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
        
        full_name = f'{coll_name}_{pass_name}'

        options = [b.GetName() for b in self.t.GetListOfBranches() if full_name in b.GetName()]

        if len(options) == 0 :
            raise Exception(f'No branch matching {full_name}.')
        elif len(options) > 1 :
            raise Exception(f'More than one branch matching {full_name}.')

        # now we know we have a unique branch 'options[0]'
        self.t.SetBranchAddress(options[0], ROOT.AddressOf(obj))

    def single_translation(self) :
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
    def __init__(self, sig_sample, bkg_sample, max_events) :

        sign_feats = sig_sample.translation(max_events = max_events, shuffle=True)
        bkgd_feats = bkg_sample.translation(max_events = max_events, shuffle=True)

        train_x = numpy.vstack((sign_feats,bkgd_feats))
        train_y = numpy.append(numpy.zeros(len(sign_feats))+1,numpy.zeros(len(bkgd_feats)))
        
        train_x[numpy.isnan(train_x)] = 0.000
        train_y[numpy.isnan(train_y)] = 0.000
        
        self.unified_training_sample = xgboost.DMatrix(train_x,train_y)

    def train(self, tree_number = 1, eta = 0.023, depth = 10) :
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
    def __init__(self, pkl_file) :
        self.model = pickle.load(open(pkl_file,'rb'))

    def __eval__(self, translated_event) :
        return float(self.model.predict(xgboost.DMatrix(numpy.array([translated_event])))[0])

    def eval(self, input_sample, out_name, max_events=None) :
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
