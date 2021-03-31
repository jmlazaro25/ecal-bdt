"""
Core methods for both training and evaluating the BDT
"""

import ROOT

import numpy
import pickle
import xgboost
import matplotlib

ROOT.gSystem.Load('libFramework.so')

class SampleContainer:
    def __init__(self, file_paths, tree_to_clone=None):
        if tree_to_clone is None :
            self.t = ROOT.TChain('LDMX_Events')
            for f in file_paths :
                self.t.Add(f)
        else :
            if len(file_paths) > 1 :
                raise Exception('Can\'t clone tree into multiple files.')

            self.f = ROOT.TFile.Open(file_paths[0],'RECREATE')
            self.t = tree_to_clone.CloneTree(0)
        # reading or writing
    
        self.veto = None

    def get_veto(self) :
        if self.veto is None :
            try :
                self.veto = self.t.EcalVeto_eat
            except :
                self.veto = self.t.EcalVeto_signal
        
        return self.veto

    def single_translation(self) :
        self.veto = self.get_veto()

        return [
            self.veto.getNReadoutHits(),
            self.veto.getSummedDet(),
            self.veto.getSummedTightIso(),
            self.veto.getMaxCellDep(),
            self.veto.getShowerRMS(),
            self.veto.getXStd(),
            self.veto.getYStd(),
            self.veto.getAvgLayerHit(),
            self.veto.getStdLayerHit(),
            self.veto.getDeepestLayerHit(),
            self.veto.getEcalBackEnergy(),
            self.veto.getNStraightTracks(),
            self.veto.getNLinRegTracks()
            ]

    def translation(self, max_events = None, shuffle = True) :
        """Translating the sample wrapped by this container into list of feature vectors"""
        events = []
        for event in self.t :
            if max_events is not None :
                max_events -= 1
                if max_events < 0 :
                    break
            events.append(self.single_translation())

        if shuffle :
            new_idx=numpy.random.permutation(numpy.arange(numpy.shape(events)[0]))
            events = numpy.array(events)
            numpy.take(events, new_idx, axis=0, out=events)

        return events

class Trainer :
    def __init__(self, sig_sample, bkg_sample, max_events, eta, depth, tree_number) :

        sign_feats = sig_sample.translation(max_events = max_events, shuffle=True)
        bkgd_feats = bkg_sample.translation(max_events = max_events, shuffle=True)

        train_x = numpy.vstack((sign_feats,bkgd_feats))
        train_y = numpy.append(numpy.zeros(len(sign_feats))+1,numpy.zeros(len(bkgd_feats)))
        
        train_x[numpy.isnan(train_x)] = 0.000
        train_y[numpy.isnan(train_y)] = 0.000
        
        self.unified_training_sample = xgboost.DMatrix(train_x,train_y)
        self.training_parameters = {'objective': 'binary:logistic',
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
        self.tree_number = tree_number

    def train(self) :
        # Actual training
        self.gbm = xgboost.train(self.training_parameters, self.unified_training_sample, self.tree_number)

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

    def eval(self, input_sample, max_events=None, out_name='eval_bdt.root') :
        output_sample = SampleContainer([out_name], input_sample.t)
        for event in input_sample.t :
            output_sample.get_veto().setDiscValue(self.__eval__(input_sample.single_translation()))
            output_sample.t.Fill()

            if max_events is not None :
                max_events -= 1
                if max_events < 0 :
                    break

        #done with loop over read-in events

        output_sample.t.Write()
        output_sample.f.Close()
