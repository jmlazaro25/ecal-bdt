import os
import sys
import logging
import argparse
import ROOT as r
import numpy as np
import pickle as pkl
import xgboost as xgb
import matplotlib as plt
from array    import array
from optparse import OptionParser


mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
plt.use('Agg')
sys.path.insert(0, '../')

# Because remembering parsing flags can be annoying sometimes
bkg_file = 'flats_1/bkg/bkg_train.root'
sig_file = 'flats_1/sig/sig_train.root'

class sampleContainer:
    def __init__(self,filename,maxEvts,isSig):

        print("Initializing Container!")
        self.tree = r.TChain("EcalVeto")
        self.tree.Add(filename)
        self.maxEvts = maxEvts
        self.isSig   = isSig

    def root2PyEvents(self):
        self.events =  []
        for event in self.tree:
            if len(self.events) >= self.maxEvts:
                continue
            evt = []

            ###################################
            # Features
            ###################################
            evt.append(event.nReadoutHits)          # 0
            evt.append(event.summedDet)             # 1
            evt.append(event.summedTightIso)        # 2
            evt.append(event.maxCellDep)            # 3
            evt.append(event.showerRMS)             # 4

            evt.append(event.xStd)                  # 5
            evt.append(event.yStd)                  # 6
            evt.append(event.avgLayerHit)           # 7
            evt.append(event.stdLayerHit)           # 8
            evt.append(event.deepestLayerHit)       # 9

            evt.append(event.ecalBackEnergy)        # 10
            evt.append(event.recoilPT)              # 11
            evt.append(event.nStraightTracks)       # 12

            self.events.append(evt)

        new_idx=np.random.permutation(np.arange(np.shape(self.events)[0]))
        self.events = np.array(self.events)
        np.take(self.events, new_idx, axis=0, out=self.events)
        print("Final Event Shape" + str(np.shape(self.events)))

    def constructTrainAndTest(self):
        self.train_x = self.events
        self.train_y = np.zeros(len(self.train_x)) + (self.isSig == True)

class mergedContainer:
    def __init__(self, sigContainer,bkgContainer):
        self.train_x = np.vstack((sigContainer.train_x,bkgContainer.train_x))
        self.train_y = np.append(sigContainer.train_y,bkgContainer.train_y)
        
        self.train_x[np.isnan(self.train_x)] = 0.000
        self.train_y[np.isnan(self.train_y)] = 0.000
        
        self.dtrain = xgb.DMatrix(self.train_x,self.train_y)
    

if __name__ == "__main__":
    
    # Parse
    parser = OptionParser()
    parser.add_option('--seed', dest='seed',type="int",  default=2, help='Numpy random seed.')
    parser.add_option('--max_evt', dest='max_evt',type="int",  default=1500000, help='Max Events to load')
    parser.add_option('--out_name', dest='out_name',  default='bdt_test', help='Output Pickle Name')
    parser.add_option('--eta', dest='eta',type="float",  default=0.023, help='Learning Rate')
    parser.add_option('--tree_number', dest='tree_number',type="int",  default=1000, help='Tree Number')
    parser.add_option('--depth', dest='depth',type="int",  default=10, help='Max Tree Depth')
    parser.add_option('--bkg_file', dest='bkg_file', default=bkg_file, help='name of background file')
    parser.add_option('--sig_file', dest='sig_file', default=sig_file, help='name of signal file')
    (options, args) = parser.parse_args()

    # Seed numpy's randomness
    np.random.seed(options.seed)
   
    # Get BDT num
    bdt_num=0
    Check=True
    while Check:
        if not os.path.exists(options.out_name+'_'+str(bdt_num)):
            try:
                os.makedirs(options.out_name+'_'+str(bdt_num))
                Check=False
            except:
               Check=True
        else:
            bdt_num+=1

    # Print run info
    print( 'Random seed is = {}'.format(options.seed)             )
    print( 'You set max_evt = {}'.format(options.max_evt)         )
    print( 'You set tree number = {}'.format(options.tree_number) )
    print( 'You set max tree depth = {}'.format(options.depth)    )
    print( 'You set eta = {}'.format(options.eta)                 )

    # Make Signal Container
    print( 'Loading sig_file = {}'.format(options.sig_file) )
    sigContainer = sampleContainer(options.sig_file,options.max_evt,True)
    sigContainer.root2PyEvents()
    sigContainer.constructTrainAndTest()

    # Make Background Container
    print( 'Loading bkg_file = {}'.format(options.bkg_file) )
    bkgContainer = sampleContainer(options.bkg_file,options.max_evt,False)
    bkgContainer.root2PyEvents()
    bkgContainer.constructTrainAndTest()

    # Merge
    eventContainer = mergedContainer(sigContainer,bkgContainer)

    params     = {'objective': 'binary:logistic',
                  'eta': options.eta,
                  'max_depth': options.depth,
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

    # Actual training
    gbm = xgb.train(params, eventContainer.dtrain, options.tree_number)

    # Store BDT
    output = open(options.out_name+'_'+str(bdt_num)+'/' + \
            options.out_name+'_'+str(bdt_num)+'_weights.pkl', 'wb')
    pkl.dump(gbm, output)

    # Plot feature importances
    xgb.plot_importance(gbm)
    plt.pyplot.savefig(options.out_name+'_'+str(bdt_num)+"/" + \
            options.out_name+'_'+str(bdt_num)+'_fimportance.png', # png file name
            dpi=500, bbox_inches='tight', pad_inches=0.5) # png parameters
    
    # Closing statment
    print("Files saved in: ", options.out_name+'_'+str(bdt_num))
