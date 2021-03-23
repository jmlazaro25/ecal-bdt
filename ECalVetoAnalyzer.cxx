#include "ECalVetoAnalyzer.h" 

/*~~~~~~~~~~*/
/*   LDMX   */
/*~~~~~~~~~~*/
#include "DetDescr/SimSpecialID.h"
#include "Event/EcalHit.h"
#include "Event/EcalVetoResult.h" 
#include "Event/EventConstants.h"
#include "Event/HcalVetoResult.h" 
#include "Event/TrackerVetoResult.h" 
#include "Event/TriggerResult.h" 
#include "Event/SimCalorimeterHit.h"
#include "Framework/NtupleManager.h"
#include "Framework/Process.h" 

/*~~~~~~~~~~~*/
/*   Tools   */
/*~~~~~~~~~~~*/
#include "Tools/AnalysisUtils.h"

/*~~~~~~~~~~*/
/*   ROOT   */
/*~~~~~~~~~~*/
#include "TMatrixD.h"
#include "TDecompSVD.h"
#include "TVector3.h"

/*~~~~~~~~~*/
/*   C++   */
/*~~~~~~~~~*/
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>

namespace ldmx { 

    // Arrays holding 68% containment radius per layer for different bins in momentum/angle
    const std::vector<double> radius68_thetalt10_plt500 = {4.045666158618167, 4.086393662224346, 4.359141107602775, 4.666549994726691, 5.8569181911416015, 6.559716356124256, 8.686967529043072, 10.063482736354674, 13.053528344041274, 14.883496407943747, 18.246694748611368, 19.939799900443724, 22.984795944506224, 25.14745829663406, 28.329169392203216, 29.468032123356345, 34.03271241527079, 35.03747443690781, 38.50748727211848, 39.41576583301171, 42.63622296033334, 45.41123601592071, 48.618139095742876, 48.11801717451056, 53.220539860213655, 58.87753380915155, 66.31550881539764, 72.94685877928593, 85.95506228335348, 89.20607201266672, 93.34370253818409, 96.59471226749734, 100.7323427930147, 103.98335252232795};

    const std::vector<double> radius68_thetalt10_pgt500 = {4.081926458777424, 4.099431732299409, 4.262428482867968, 4.362017581473145, 4.831341579961153, 4.998346041276382, 6.2633736512415705, 6.588371889265881, 8.359969947444522, 9.015085558044309, 11.262722588206483, 12.250305471269183, 15.00547660437276, 16.187264014640103, 19.573764900578503, 20.68072032434797, 24.13797140783321, 25.62942209291236, 29.027596514735617, 30.215039667389316, 33.929540248019585, 36.12911729771914, 39.184563500620946, 42.02062468386282, 46.972125628650204, 47.78214816041894, 55.88428562462974, 59.15520134927332, 63.31816666637158, 66.58908239101515, 70.75204770811342, 74.022963432757, 78.18592874985525, 81.45684447449884};

    const std::vector<double> radius68_theta10to20 = {4.0251896715647115, 4.071661598616328, 4.357690094817289, 4.760224640141712, 6.002480766325418, 6.667318981016246, 8.652513285172342, 9.72379373302137, 12.479492693251478, 14.058548828317289, 17.544872909347912, 19.43616066939176, 23.594162859513734, 25.197329065282954, 29.55995803074302, 31.768946746958296, 35.79247330197688, 37.27810357669942, 41.657281051476545, 42.628141392692626, 47.94208483539388, 49.9289473559796, 54.604030254423975, 53.958762417361655, 53.03339560920388, 57.026277390001425, 62.10810455035879, 66.10098633115634, 71.1828134915137, 75.17569527231124, 80.25752243266861, 84.25040421346615, 89.33223137382352, 93.32511315462106};

    const std::vector<double> radius68_thetagt20 = {4.0754238481177705, 4.193693485630508, 5.14209420056253, 6.114996249971468, 7.7376807326481645, 8.551663213602291, 11.129110612057813, 13.106293737495639, 17.186617323282082, 19.970887612094604, 25.04088272634407, 28.853696411302344, 34.72538105333071, 40.21218694947545, 46.07344239520299, 50.074953583805346, 62.944045771758645, 61.145621459396814, 69.86940198299047, 74.82378572939959, 89.4528387422834, 93.18228303096758, 92.51751129204555, 98.80228884380018, 111.17537347472128, 120.89712563907408, 133.27021026999518, 142.99196243434795, 155.36504706526904, 165.08679922962185, 177.45988386054293, 187.18163602489574, 199.55472065581682, 209.2764728201696};

    // List of all layer z positions
    const std::vector<double> LAYER_Z_POSITIONS = {223.8000030517578, 226.6999969482422, 233.0500030517578, 237.4499969482422, 245.3000030517578, 251.1999969482422, 260.29998779296875, 266.70001220703125, 275.79998779296875, 282.20001220703125, 291.29998779296875, 297.70001220703125, 306.79998779296875, 313.20001220703125, 322.29998779296875, 328.70001220703125, 337.79998779296875, 344.20001220703125, 353.29998779296875, 359.70001220703125, 368.79998779296875, 375.20001220703125, 384.29998779296875, 390.70001220703125, 403.29998779296875, 413.20001220703125, 425.79998779296875, 435.70001220703125, 448.29998779296875, 458.20001220703125, 470.79998779296875, 480.70001220703125, 493.29998779296875, 503.20001220703125};

    // Starting/stopping points for longitudinal segments
    const std::vector<int> segs = {0, 6, 17, 34};

    // ECal parameters
    const int nECalLayers = 34;
    const int nregions = 5;
    const int nsegments = 3;

    ECalVetoAnalyzer::ECalVetoAnalyzer(const std::string &name, Process &process) : 
        Analyzer(name, process) {
    }

    ECalVetoAnalyzer::~ECalVetoAnalyzer() { 
    }

    void ECalVetoAnalyzer::onProcessStart() {
        getHistoDirectory();

        ntuple_.create("EcalVeto"); 

        // Gabrielle and misc. variables
        ntuple_.addVar<int>("EcalVeto", "trigPass");  
        ntuple_.addVar<int>("EcalVeto", "passTrackerVeto");
        ntuple_.addVar<float>("EcalVeto", "hcalMaxPE");
        ntuple_.addVar<int>("EcalVeto", "passHcalVeto");
        ntuple_.addVar<int>("EcalVeto", "nReadoutHits");  
        ntuple_.addVar<float>("EcalVeto", "summedDet");
        ntuple_.addVar<float>("EcalVeto", "summedTightIso");
        ntuple_.addVar<float>("EcalVeto", "maxCellDep");
        ntuple_.addVar<float>("EcalVeto", "showerRMS");
        ntuple_.addVar<float>("EcalVeto", "xStd");
        ntuple_.addVar<float>("EcalVeto", "yStd");
        ntuple_.addVar<float>("EcalVeto", "avgLayerHit");
        ntuple_.addVar<float>("EcalVeto", "stdLayerHit");
        ntuple_.addVar<int>("EcalVeto", "deepestLayerHit");
        ntuple_.addVar<float>("EcalVeto", "ecalBackEnergy");
        ntuple_.addVar<float>("EcalVeto", "discValue"); 
        ntuple_.addVar<int>("EcalVeto", "nNoiseHits");  
        ntuple_.addVar<float>("EcalVeto", "noiseEnergy");  
        ntuple_.addVar<float>("EcalVeto", "trigEnergy");
        ntuple_.addVar<int>("EcalVeto", "nRecHits");
        ntuple_.addVar<int>("EcalVeto", "nSimHits");
        ntuple_.addVar<int>("EcalVeto", "nElectrons");
        ntuple_.addVar<double>("EcalVeto", "recoilPT");

        // Segmentation variables for total regions
        for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
            ntuple_.addVar<float>("EcalVeto", "totalEnergy_l" + std::to_string(ireg + 1));
        }
        for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
            ntuple_.addVar<int>("EcalVeto", "totalNHits_l" + std::to_string(ireg + 1));
        }
        for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
            ntuple_.addVar<float>("EcalVeto", "totalXMean_l" + std::to_string(ireg + 1));
        }
        for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
            ntuple_.addVar<float>("EcalVeto", "totalYMean_l" + std::to_string(ireg + 1));
        }
        for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
            ntuple_.addVar<float>("EcalVeto", "totalZMean_l" + std::to_string(ireg + 1));
        }
        for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
            ntuple_.addVar<float>("EcalVeto", "totalXStd_l" + std::to_string(ireg + 1));
        }
        for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
            ntuple_.addVar<float>("EcalVeto", "totalYStd_l" + std::to_string(ireg + 1));
        }
        for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
            ntuple_.addVar<float>("EcalVeto", "totalZStd_l" + std::to_string(ireg + 1));
        }

        // Segmentation variables for electron containment regions
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "electronContainmentEnergy_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<int>("EcalVeto", "electronContainmentNHits_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "electronContainmentXMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "electronContainmentYMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "electronContainmentZMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "electronContainmentXStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "electronContainmentYStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "electronContainmentZStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }

        // Segmentation variables for photon containment regions
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "photonContainmentEnergy_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<int>("EcalVeto", "photonContainmentNHits_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "photonContainmentXMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "photonContainmentYMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "photonContainmentZMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "photonContainmentXStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "photonContainmentYStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "photonContainmentZStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }

        // Segmentation variables for regions outside of the electron/photon containment regions
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "outsideContainmentEnergy_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<int>("EcalVeto", "outsideContainmentNHits_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "outsideContainmentXMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "outsideContainmentYMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "outsideContainmentZMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "outsideContainmentXStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "outsideContainmentYStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                ntuple_.addVar<float>("EcalVeto", "outsideContainmentZStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1));
            }
        }

        // MIP tracking variables
        ntuple_.addVar<int>("EcalVeto", "nStraightTracks");
        ntuple_.addVar<int>("EcalVeto", "nLinregTracks");
        ntuple_.addVar<int>("EcalVeto", "firstNearPhLayer");
        ntuple_.addVar<float>("EcalVeto", "epAng");
        ntuple_.addVar<float>("EcalVeto", "epSep");

        hitTree_ = new TTree("EcalHits", "EcalHits");
        hitTree_->Branch("hitX", &hitXv);
        hitTree_->Branch("hitY", &hitYv);
        hitTree_->Branch("hitZ", &hitZv);
        hitTree_->Branch("hitLayer", &hitLayerv);
        hitTree_->Branch("recHitEnergy", &recHitEnergyv);
        hitTree_->Branch("recHitAmplitude", &recHitAmplitudev);
        hitTree_->Branch("simHitEnergy", &simHitMatchEnergyv);
        hitTree_->Branch("trigPass", &trigPass, "trigPass/I");  

        simHitTree_ = new TTree("EcalSimHits", "EcalSimHits");
        simHitTree_->Branch("hitX", &simHitXv);
        simHitTree_->Branch("hitY", &simHitYv);
        simHitTree_->Branch("hitZ", &simHitZv);
        simHitTree_->Branch("hitLayer", &simHitLayerv);
        simHitTree_->Branch("simHitEnergy", &simHitEnergyv);
    }

    void ECalVetoAnalyzer::configure(Parameters& parameters) {

        // Set the name of the collections to use
        trigResultCollectionName_ = parameters.getParameter< std::string>("trig_result_collection"); 
        trackerVetoCollectionName_ = parameters.getParameter< std::string>("tracker_veto_collection"); 
        hcalVetoCollectionName_ = parameters.getParameter< std::string>("hcal_veto_collection"); 
        ecalVetoCollectionName_ = parameters.getParameter< std::string>("ecal_veto_collection"); 
        ecalSimHitCollectionName_ = parameters.getParameter< std::string>("ecal_simhit_collection"); 
        ecalRecHitCollectionName_ = parameters.getParameter< std::string>("ecal_rechit_collection"); 
        simParticleCollectionName_ = parameters.getParameter< std::string>("sim_particle_collection"); 

        // Loaded in during veto analysis
        hexReadout_ = 0;

    }

    void ECalVetoAnalyzer::analyze(const Event& event) {

        hitXv.clear();
        hitYv.clear();
        hitZv.clear();
        hitLayerv.clear();
        recHitEnergyv.clear();
        recHitAmplitudev.clear();
        simHitMatchEnergyv.clear();
        simHitXv.clear();
        simHitYv.clear();
        simHitZv.clear();
        simHitLayerv.clear();
        simHitEnergyv.clear();

        const EcalHexReadout &ecalHexReadout = getCondition<EcalHexReadout>(EcalHexReadout::CONDITIONS_OBJECT_NAME);
        hexReadout_ = &ecalHexReadout;

        trigPass = 1;
   
        // Check for the ECal SimHit collection in the event.  If it doesn't 
        // exist, skip creating an ntuple.

        // Set variables; check first that the necessary collections exist
        if(event.exists(trigResultCollectionName_)) {
            auto trigResult{event.getObject<TriggerResult>(trigResultCollectionName_)};
            trigPass = trigResult.passed();
            ntuple_.setVar<int>("trigPass", trigPass);  
        }

        if(event.exists(trackerVetoCollectionName_)) {
            auto trackerVeto{event.getObject<TrackerVetoResult>(trackerVetoCollectionName_)};
            ntuple_.setVar<int>("passTrackerVeto", trackerVeto.passesVeto());
        }

        if(event.exists(hcalVetoCollectionName_)) {
            auto hcalVeto{event.getObject<HcalVetoResult>(hcalVetoCollectionName_)};
            ntuple_.setVar<float>("hcalMaxPE", hcalVeto.getMaxPEHit().getPE());
            ntuple_.setVar<int>("passHcalVeto", hcalVeto.passesVeto());
        }

        if(event.exists(ecalVetoCollectionName_)) {
            auto ecalVeto{event.getObject<EcalVetoResult>(ecalVetoCollectionName_)};
            ntuple_.setVar<int>("nReadoutHits", ecalVeto.getNReadoutHits());  
            ntuple_.setVar<float>("summedDet", ecalVeto.getSummedDet());
            ntuple_.setVar<float>("summedTightIso", ecalVeto.getSummedTightIso());
            ntuple_.setVar<float>("maxCellDep", ecalVeto.getMaxCellDep());
            ntuple_.setVar<float>("showerRMS", ecalVeto.getShowerRMS());
            ntuple_.setVar<float>("xStd", ecalVeto.getXStd());
            ntuple_.setVar<float>("yStd", ecalVeto.getYStd());
            ntuple_.setVar<float>("avgLayerHit", ecalVeto.getAvgLayerHit());
            ntuple_.setVar<float>("stdLayerHit", ecalVeto.getStdLayerHit());
            ntuple_.setVar<int>("deepestLayerHit", ecalVeto.getDeepestLayerHit());
            ntuple_.setVar<float>("ecalBackEnergy", ecalVeto.getEcalBackEnergy());
            ntuple_.setVar<float>("discValue", ecalVeto.getDisc());  
        }

        std::vector<double> recoilP;
        std::vector<float> recoilPos;
        std::vector<double> recoilPAtTarget;
        std::vector<float> recoilPosAtTarget;
        double recoilPT;

        if(event.exists("EcalScoringPlaneHits")) {

            // Loop through all of the sim particles and find the recoil electron

            // Get the collection of simulated particles from the event
            auto particleMap{event.getMap<int, SimParticle>("SimParticles")};

            // Search for the recoil electron
            auto [recoilTrackID, recoilElectron] = Analysis::getRecoil(particleMap);

            // Find ECAL SP hit for recoil electron
            auto ecalSpHits{event.getCollection<SimTrackerHit>("EcalScoringPlaneHits")};
            float pmax = 0;
            for(SimTrackerHit &spHit : ecalSpHits) {
                SimSpecialID hit_id(spHit.getID());
                if(hit_id.plane() != 31 || spHit.getMomentum()[2] <= 0) continue;
                if(spHit.getTrackID() == recoilTrackID) {
                    if(sqrt(pow(spHit.getMomentum()[0], 2) + pow(spHit.getMomentum()[1], 2) + pow(spHit.getMomentum()[2], 2)) > pmax) {
                        recoilP = spHit.getMomentum();
                        recoilPos = spHit.getPosition();
                        pmax = sqrt(pow(recoilP[0], 2) + pow(recoilP[1], 2) + pow(recoilP[2], 2));
                    }
                }
            }

            // Find target SP hit for recoil electron
            if(event.exists("TargetScoringPlaneHits")) {
                std::vector<SimTrackerHit> targetSpHits = event.getCollection<SimTrackerHit>("TargetScoringPlaneHits");
                pmax = 0;
                for(SimTrackerHit &spHit : targetSpHits) {
                    SimSpecialID hit_id(spHit.getID());
                    if(hit_id.plane() != 1 || spHit.getMomentum()[2] <= 0) continue;
                    if(spHit.getTrackID() == recoilTrackID) {
                        if(sqrt(pow(spHit.getMomentum()[0], 2) + pow(spHit.getMomentum()[1], 2) + pow(spHit.getMomentum()[2], 2)) > pmax) {
                            recoilPAtTarget = spHit.getMomentum();
                            recoilPosAtTarget = spHit.getPosition();
                            pmax = sqrt(pow(recoilPAtTarget[0], 2) + pow(recoilPAtTarget[1], 2) + pow(recoilPAtTarget[2], 2));
                        }
                    }
                }
            }

            if(recoilPAtTarget.size() > 0) {
                recoilPT = sqrt(pow(recoilPAtTarget[0], 2) + pow(recoilPAtTarget[1], 2));
                ntuple_.setVar<double>("recoilPT", recoilPT);
            }

        }

        // Segmentation variables for total regions
        std::vector<float> totalEnergy (nsegments, 0.0);
        std::vector<int> totalNHits (nsegments, 0);
        std::vector<float> totalXMean (nsegments, 0.0);
        std::vector<float> totalYMean (nsegments, 0.0);
        std::vector<float> totalZMean (nsegments, 0.0);
        std::vector<float> totalXStd (nsegments, 0.0);
        std::vector<float> totalYStd (nsegments, 0.0);
        std::vector<float> totalZStd (nsegments, 0.0);

        // Segmentation variables for electron containment regions
        std::vector<std::vector<float>> electronContainmentEnergy;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            electronContainmentEnergy.push_back(vec);
        }
        std::vector<std::vector<int>> electronContainmentNHits;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<int> vec (nsegments, 0);
            electronContainmentNHits.push_back(vec);
        }
        std::vector<std::vector<float>> electronContainmentXMean;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            electronContainmentXMean.push_back(vec);
        }
        std::vector<std::vector<float>> electronContainmentYMean;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            electronContainmentYMean.push_back(vec);
        }
        std::vector<std::vector<float>> electronContainmentZMean;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            electronContainmentZMean.push_back(vec);
        }
        std::vector<std::vector<float>> electronContainmentXStd;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            electronContainmentXStd.push_back(vec);
        }
        std::vector<std::vector<float>> electronContainmentYStd;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            electronContainmentYStd.push_back(vec);
        }
        std::vector<std::vector<float>> electronContainmentZStd;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            electronContainmentZStd.push_back(vec);
        }

        // Segmentation variables for photon containment regions
        std::vector<std::vector<float>> photonContainmentEnergy;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            photonContainmentEnergy.push_back(vec);
        }
        std::vector<std::vector<int>> photonContainmentNHits;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<int> vec (nsegments, 0);
            photonContainmentNHits.push_back(vec);
        }
        std::vector<std::vector<float>> photonContainmentXMean;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            photonContainmentXMean.push_back(vec);
        }
        std::vector<std::vector<float>> photonContainmentYMean;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            photonContainmentYMean.push_back(vec);
        }
        std::vector<std::vector<float>> photonContainmentZMean;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            photonContainmentZMean.push_back(vec);
        }
        std::vector<std::vector<float>> photonContainmentXStd;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            photonContainmentXStd.push_back(vec);
        }
        std::vector<std::vector<float>> photonContainmentYStd;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            photonContainmentYStd.push_back(vec);
        }
        std::vector<std::vector<float>> photonContainmentZStd;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            photonContainmentZStd.push_back(vec);
        }

        // Segmentation variables for regions outside of electron/photon containment regions
        std::vector<std::vector<float>> outsideContainmentEnergy;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            outsideContainmentEnergy.push_back(vec);
        }
        std::vector<std::vector<int>> outsideContainmentNHits;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<int> vec (nsegments, 0);
            outsideContainmentNHits.push_back(vec);
        }
        std::vector<std::vector<float>> outsideContainmentXMean;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            outsideContainmentXMean.push_back(vec);
        }
        std::vector<std::vector<float>> outsideContainmentYMean;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            outsideContainmentYMean.push_back(vec);
        }
        std::vector<std::vector<float>> outsideContainmentZMean;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            outsideContainmentZMean.push_back(vec);
        }
        std::vector<std::vector<float>> outsideContainmentXStd;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            outsideContainmentXStd.push_back(vec);
        }
        std::vector<std::vector<float>> outsideContainmentYStd;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            outsideContainmentYStd.push_back(vec);
        }
        std::vector<std::vector<float>> outsideContainmentZStd;
        for(unsigned int ireg = 0; ireg < nregions; ireg++) {
            std::vector<float> vec (nsegments, 0.0);
            outsideContainmentZStd.push_back(vec);
        }
 
        // MIP tracking variables
        int nStraightTracks = 0;
        int nLinregTracks = 0;
        int firstNearPhLayer = 0;
        float epAng = 0.0;
        float epSep = 0.0;

        if(event.exists(ecalRecHitCollectionName_)) {

            // Get projected trajectories for electron and photon
            std::vector<XYCoords> ele_trajectory, photon_trajectory;
            if(recoilP.size() > 0) {
                ele_trajectory = getTrajectory(recoilP, recoilPos);
                std::vector<double> pvec = recoilPAtTarget.size() ? recoilPAtTarget : std::vector<double>{0.0, 0.0, 0.0};
                std::vector<float>  posvec = recoilPosAtTarget.size() ? recoilPosAtTarget : std::vector<float>{0.0, 0.0, 0.0};
                photon_trajectory = getTrajectory({-pvec[0], -pvec[1], 4000.0 - pvec[2]}, posvec);
            }

            float recoilPMag = recoilP.size() ? sqrt(pow(recoilP[0], 2) + pow(recoilP[1], 2) + pow(recoilP[2], 2)) : -1.0;
            float recoilTheta = recoilPMag > 0 ? recoilP[2]/recoilPMag : -1.0;

            // Set the binning to use for the electron radii of containment
            std::vector<double> ele_radii = radius68_thetalt10_plt500;
            if(recoilTheta < 10 && recoilPMag >= 500) ele_radii = radius68_thetalt10_pgt500;
            else if(recoilTheta >= 10 && recoilTheta < 20) ele_radii = radius68_theta10to20;
            else if(recoilTheta >= 20) ele_radii = radius68_thetagt20;

            // Always use default binning for the photon radii of containment?
            std::vector<double> photon_radii = radius68_thetalt10_plt500;

            // Get the collection of digitized Ecal hits from the event
            const std::vector<EcalHit> ecalRecHits = event.getCollection<EcalHit>(ecalRecHitCollectionName_);

            // MIP tracking hit list
            std::vector<HitData> trackingHitList;

            for(const EcalHit &hit : ecalRecHits) {
                EcalID id = hitID(hit);
                if(hit.getEnergy() > 0) {
                    XYCoords xy_pair = getCellCentroidXYPair(id);
                    float distance_ele_trajectory = ele_trajectory.size() ? sqrt(pow((xy_pair.first - ele_trajectory[id.layer()].first), 2)
                                                    + pow((xy_pair.second - ele_trajectory[id.layer()].second), 2)) : -1.0;
                    float distance_photon_trajectory = photon_trajectory.size() ? sqrt(pow((xy_pair.first - photon_trajectory[id.layer()].first), 2)
                                                       + pow((xy_pair.second - photon_trajectory[id.layer()].second), 2)) : -1.0;

                    // Decide which region a hit is in and add to sums
                    for(int ireg = 0; ireg < nsegments; ireg++) {
                        if(id.layer() >= segs[ireg] && id.layer() <= segs[ireg + 1] - 1) {
                            totalEnergy[ireg] += hit.getEnergy();
                            totalNHits[ireg] += 1;
                            totalXMean[ireg] += xy_pair.first*hit.getEnergy();
                            totalYMean[ireg] += xy_pair.second*hit.getEnergy();
                            totalZMean[ireg] += id.layer()*hit.getEnergy();
                        }
                    }

                    for(int ireg = 0; ireg < nregions; ireg++) {
                        if(distance_ele_trajectory >= ireg*ele_radii[id.layer()] && distance_ele_trajectory < (ireg + 1)*ele_radii[id.layer()]) {
                            for(int jreg = 0; jreg < nsegments; jreg++) {
                                if(id.layer() >= segs[jreg] && id.layer() <= segs[jreg + 1] - 1) {
                                    electronContainmentEnergy[ireg][jreg] += hit.getEnergy();
                                    electronContainmentNHits[ireg][jreg] += 1;
                                    electronContainmentXMean[ireg][jreg] += xy_pair.first*hit.getEnergy();
                                    electronContainmentYMean[ireg][jreg] += xy_pair.second*hit.getEnergy();
                                    electronContainmentZMean[ireg][jreg] += id.layer()*hit.getEnergy();
                                }
                            }
                        }
                        if(distance_photon_trajectory >= ireg*photon_radii[id.layer()] && distance_photon_trajectory < (ireg+1)*photon_radii[id.layer()]) {
                            for(int jreg = 0; jreg < nsegments; jreg++) {
                                if(id.layer() >= segs[jreg] && id.layer() <= segs[jreg + 1] - 1) {
                                    photonContainmentEnergy[ireg][jreg] += hit.getEnergy();
                                    photonContainmentNHits[ireg][jreg] += 1;
                                    photonContainmentXMean[ireg][jreg] += xy_pair.first*hit.getEnergy();
                                    photonContainmentYMean[ireg][jreg] += xy_pair.second*hit.getEnergy();
                                    photonContainmentZMean[ireg][jreg] += id.layer()*hit.getEnergy();
                                }
                            }
                        }
                        if(distance_ele_trajectory > (ireg + 1)*ele_radii[id.layer()] && distance_photon_trajectory > (ireg + 1)*photon_radii[id.layer()]) {
                            for(int jreg = 0; jreg < nsegments; jreg++) {
                                if(id.layer() >= segs[jreg] && id.layer() <= segs[jreg + 1] - 1) {
                                    outsideContainmentEnergy[ireg][jreg] += hit.getEnergy();
                                    outsideContainmentNHits[ireg][jreg] += 1;
                                    outsideContainmentXMean[ireg][jreg] += xy_pair.first*hit.getEnergy();
                                    outsideContainmentYMean[ireg][jreg] += xy_pair.second*hit.getEnergy();
                                    outsideContainmentZMean[ireg][jreg] += id.layer()*hit.getEnergy();
                                }
                            }
                        }
                    }

                    // Decide whether hit should be added to the hit list: Add if inside electron containment region or if trajectory is missing
                    if(distance_ele_trajectory >= ele_radii[id.layer()] || distance_ele_trajectory == -1.0) {
                        HitData hd;
                        hd.pos = TVector3(xy_pair.first, xy_pair.second, LAYER_Z_POSITIONS[id.layer()]);
                        hd.layer = id.layer();
                        trackingHitList.push_back(hd);
                    }
                }
            }

            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                if(totalEnergy[ireg] > 0) {
                    totalXMean[ireg] /= totalEnergy[ireg];
                    totalYMean[ireg] /= totalEnergy[ireg];
                    totalZMean[ireg] /= totalEnergy[ireg];
                }
            }

            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                    if(electronContainmentEnergy[ireg][jreg] > 0) {
                        electronContainmentXMean[ireg][jreg] /= electronContainmentEnergy[ireg][jreg];
                        electronContainmentYMean[ireg][jreg] /= electronContainmentEnergy[ireg][jreg];
                        electronContainmentZMean[ireg][jreg] /= electronContainmentEnergy[ireg][jreg];
                    }
                    if(photonContainmentEnergy[ireg][jreg] > 0) {
                        photonContainmentXMean[ireg][jreg] /= photonContainmentEnergy[ireg][jreg];
                        photonContainmentYMean[ireg][jreg] /= photonContainmentEnergy[ireg][jreg];
                        photonContainmentZMean[ireg][jreg] /= photonContainmentEnergy[ireg][jreg];
                    }
                    if(outsideContainmentEnergy[ireg][jreg] > 0) {
                        outsideContainmentXMean[ireg][jreg] /= outsideContainmentEnergy[ireg][jreg];
                        outsideContainmentYMean[ireg][jreg] /= outsideContainmentEnergy[ireg][jreg];
                        outsideContainmentZMean[ireg][jreg] /= outsideContainmentEnergy[ireg][jreg];
                    }
                }
            }

            // Loop over hits a second time to find the standard deviations
            for(const EcalHit &hit : ecalRecHits) {
                EcalID id = hitID(hit);
                XYCoords xy_pair = getCellCentroidXYPair(id);
                float distance_ele_trajectory = ele_trajectory.size() ? sqrt(pow((xy_pair.first - ele_trajectory[id.layer()].first), 2)
                                                + pow((xy_pair.second - ele_trajectory[id.layer()].second), 2)) : -1.0;
                float distance_photon_trajectory = photon_trajectory.size() ? sqrt(pow((xy_pair.first - photon_trajectory[id.layer()].first), 2)
                                                   + pow((xy_pair.second - photon_trajectory[id.layer()].second), 2)) : -1.0;
                for(int ireg = 0; ireg < nsegments; ireg++) {
                    if(id.layer() >= segs[ireg] && id.layer() <= segs[ireg + 1] - 1) {
                        totalXStd[ireg] += pow((xy_pair.first - totalXMean[ireg]), 2)*hit.getEnergy();
                        totalYStd[ireg] += pow((xy_pair.second - totalYMean[ireg]), 2)*hit.getEnergy();
                        totalZStd[ireg] += pow((id.layer() - totalZMean[ireg]), 2)*hit.getEnergy();
                    } 
                }
                for(int ireg = 0; ireg < nregions; ireg++) {
                    if(distance_ele_trajectory >= ireg*ele_radii[id.layer()] && distance_ele_trajectory < (ireg + 1)*ele_radii[id.layer()]) {
                        for(int jreg = 0; jreg < nsegments; jreg++) {
                            if(id.layer() >= segs[jreg] && id.layer() <= segs[jreg + 1] - 1) {
                                electronContainmentXStd[ireg][jreg] += pow((xy_pair.first - electronContainmentXMean[ireg][jreg]), 2)*hit.getEnergy();
                                electronContainmentYStd[ireg][jreg] += pow((xy_pair.second - electronContainmentYMean[ireg][jreg]), 2)*hit.getEnergy();
                                electronContainmentZStd[ireg][jreg] += pow((id.layer() - electronContainmentZMean[ireg][jreg]), 2)*hit.getEnergy();
                            }
                        }
                    }
                    if(distance_photon_trajectory >= ireg*photon_radii[id.layer()] && distance_photon_trajectory < (ireg + 1)*photon_radii[id.layer()]) {
                        for(int jreg = 0; jreg < nsegments; jreg++) {
                            if(id.layer() >= segs[jreg] && id.layer() <= segs[jreg + 1] - 1) {
                                photonContainmentXStd[ireg][jreg] += pow((xy_pair.first - photonContainmentXMean[ireg][jreg]), 2)*hit.getEnergy();
                                photonContainmentYStd[ireg][jreg] += pow((xy_pair.second - photonContainmentYMean[ireg][jreg]), 2)*hit.getEnergy();
                                photonContainmentZStd[ireg][jreg] += pow((id.layer() - photonContainmentZMean[ireg][jreg]), 2)*hit.getEnergy();
                            }
                        }
                    }
                    if(distance_ele_trajectory > (ireg + 1)*ele_radii[id.layer()] && distance_photon_trajectory > (ireg + 1)*photon_radii[id.layer()]) {
                        for(int jreg = 0; jreg < nsegments; jreg++) {
                            if(id.layer() >= segs[jreg] && id.layer() <= segs[jreg + 1] - 1) {
                                outsideContainmentXStd[ireg][jreg] += pow((xy_pair.first - outsideContainmentXMean[ireg][jreg]), 2)*hit.getEnergy();
                                outsideContainmentYStd[ireg][jreg] += pow((xy_pair.second - outsideContainmentYMean[ireg][jreg]), 2)*hit.getEnergy();
                                outsideContainmentZStd[ireg][jreg] += pow((id.layer() - outsideContainmentZMean[ireg][jreg]), 2)*hit.getEnergy();
                            }
                        }
                    }
                }
            }

            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                if(totalEnergy[ireg] > 0) {
                    totalXStd[ireg] = sqrt(totalXStd[ireg]/totalEnergy[ireg]);
                    totalYStd[ireg] = sqrt(totalYStd[ireg]/totalEnergy[ireg]);
                    totalZStd[ireg] = sqrt(totalZStd[ireg]/totalEnergy[ireg]);
                }
            }

            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg++) {
                    if(electronContainmentEnergy[ireg][jreg] > 0) {
                        electronContainmentXStd[ireg][jreg] = sqrt(electronContainmentXStd[ireg][jreg]/electronContainmentEnergy[ireg][jreg]);
                        electronContainmentYStd[ireg][jreg] = sqrt(electronContainmentYStd[ireg][jreg]/electronContainmentEnergy[ireg][jreg]);
                        electronContainmentZStd[ireg][jreg] = sqrt(electronContainmentZStd[ireg][jreg]/electronContainmentEnergy[ireg][jreg]);
                    }
                    if(photonContainmentEnergy[ireg][jreg] > 0) {
                        photonContainmentXStd[ireg][jreg] = sqrt(photonContainmentXStd[ireg][jreg]/photonContainmentEnergy[ireg][jreg]);
                        photonContainmentYStd[ireg][jreg] = sqrt(photonContainmentYStd[ireg][jreg]/photonContainmentEnergy[ireg][jreg]);
                        photonContainmentZStd[ireg][jreg] = sqrt(photonContainmentZStd[ireg][jreg]/photonContainmentEnergy[ireg][jreg]);
                    }
                    if(outsideContainmentEnergy[ireg][jreg] > 0) {
                        outsideContainmentXStd[ireg][jreg] = sqrt(outsideContainmentXStd[ireg][jreg]/outsideContainmentEnergy[ireg][jreg]);
                        outsideContainmentYStd[ireg][jreg] = sqrt(outsideContainmentYStd[ireg][jreg]/outsideContainmentEnergy[ireg][jreg]);
                        outsideContainmentZStd[ireg][jreg] = sqrt(outsideContainmentZStd[ireg][jreg]/outsideContainmentEnergy[ireg][jreg]);
                    }
                }
            }

            // Segmentation variables for total regions
            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                ntuple_.setVar<float>("totalEnergy_l" + std::to_string(ireg + 1), totalEnergy[ireg]);
            }
            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                ntuple_.setVar<int>("totalNHits_l" + std::to_string(ireg + 1), totalNHits[ireg]);
            }
            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                ntuple_.setVar<float>("totalXMean_l" + std::to_string(ireg + 1), totalXMean[ireg]);
            }
            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                ntuple_.setVar<float>("totalYMean_l" + std::to_string(ireg + 1), totalYMean[ireg]);
            }
            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                ntuple_.setVar<float>("totalZMean_l" + std::to_string(ireg + 1), totalZMean[ireg]);
            }
            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                ntuple_.setVar<float>("totalXStd_l" + std::to_string(ireg + 1), totalXStd[ireg]);
            }
            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                ntuple_.setVar<float>("totalYStd_l" + std::to_string(ireg + 1), totalYStd[ireg]);
            }
            for(unsigned int ireg = 0; ireg < nsegments; ireg++) {
                ntuple_.setVar<float>("totalZStd_l" + std::to_string(ireg + 1), totalZStd[ireg]);
            }

            // Segmentation variables for electron containment regions
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("electronContainmentEnergy_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), electronContainmentEnergy[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<int>("electronContainmentNHits_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), electronContainmentNHits[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("electronContainmentXMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), electronContainmentXMean[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("electronContainmentYMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), electronContainmentYMean[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("electronContainmentZMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), electronContainmentZMean[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("electronContainmentXStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), electronContainmentXStd[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("electronContainmentYStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), electronContainmentYStd[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("electronContainmentZStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), electronContainmentZStd[ireg][jreg]);
               }
            }

            // Segmentation variables for photon containment regions
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("photonContainmentEnergy_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), photonContainmentEnergy[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<int>("photonContainmentNHits_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), photonContainmentNHits[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("photonContainmentXMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), photonContainmentXMean[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("photonContainmentYMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), photonContainmentYMean[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("photonContainmentZMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), photonContainmentZMean[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("photonContainmentXStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), photonContainmentXStd[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("photonContainmentYStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), photonContainmentYStd[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("photonContainmentZStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), photonContainmentZStd[ireg][jreg]);
                }
            }

            // Segmentation variables for regions outside of the electron/photon regions
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("outsideContainmentEnergy_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), outsideContainmentEnergy[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<int>("outsideContainmentNHits_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), outsideContainmentNHits[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("outsideContainmentXMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), outsideContainmentXMean[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("outsideContainmentYMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), outsideContainmentYMean[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("outsideContainmentZMean_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), outsideContainmentZMean[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("outsideContainmentXStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), outsideContainmentXStd[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("outsideContainmentYStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), outsideContainmentYStd[ireg][jreg]);
                }
            }
            for(unsigned int ireg = 0; ireg < nregions; ireg++) {
                for(unsigned int jreg = 0; jreg < nsegments; jreg ++) {
                    ntuple_.setVar<float>("outsideContainmentZStd_x" + std::to_string(ireg + 1) + "_l" + std::to_string(jreg + 1), outsideContainmentZStd[ireg][jreg]);
                }
            }

            // MIP tracking starts here

            /* Goal: Calculate 
             *  nStraightTracks (Self-explanatory) 
             *  nLinregTracks (Tracks found by linreg algorithm)
             */

            // Find epAng and epSep, and prepare EP trajectory vectors
            TVector3 e_traj_start;
            TVector3 e_traj_end;
            TVector3 p_traj_start;
            TVector3 p_traj_end;

            if(ele_trajectory.size() > 0 && photon_trajectory.size() > 0) {

                // Create TVector3s marking the start and endpoints of each projected trajectory
                e_traj_start.SetXYZ(ele_trajectory[0].first, ele_trajectory[0].second, LAYER_Z_POSITIONS.front());
                e_traj_end.SetXYZ(ele_trajectory[33].first, ele_trajectory[33].second, LAYER_Z_POSITIONS.back());
                p_traj_start.SetXYZ(photon_trajectory[0].first, photon_trajectory[0].second, LAYER_Z_POSITIONS.front());
                p_traj_end.SetXYZ(photon_trajectory[33].first, photon_trajectory[33].second, LAYER_Z_POSITIONS.back());

                TVector3 evec = e_traj_end - e_traj_start;
                TVector3 e_norm = evec.Unit();
                TVector3 pvec = p_traj_end - p_traj_start;
                TVector3 p_norm = pvec.Unit();

                // Separation variables are currently unused due to pT bias concerns and low efficiency
                // May be removed after a more careful MIP tracking study
                float epDot = e_norm.Dot(p_norm);
                epAng = acos(epDot)*180.0/M_PI;
                epSep = sqrt(pow(e_traj_start.X() - p_traj_start.X(), 2) + pow(e_traj_start.Y() - p_traj_start.Y(), 2));
            }

            else {

                // Electron trajectory is missing, so all hits in the Ecal are fair game
                // Pick electron and photon trajectories so that they won't restrict the tracking algorithm (Place them far outside the ECal)
                e_traj_start = TVector3(999, 999, 0);
                e_traj_end = TVector3(999, 999, 999);
                p_traj_start = TVector3(1000, 1000, 0);
                p_traj_end = TVector3(1000, 1000, 1000);
                epAng = 3.0 + 1.0; // Ensures event will not be vetoed by angle/separation cut
                epSep = 10.0 + 1.0;
            }

            // Near photon step: Find the first layer of the ECal where a hit near the projected photon trajectory is found
            // Currently unusued pending further study; performance has dropped between v9 and v12
            firstNearPhLayer = 33;

            if(photon_trajectory.size() != 0) { //If no photon trajectory, leave this at the default (ECal back)
                for(std::vector<HitData>::iterator it = trackingHitList.begin(); it != trackingHitList.end(); ++it) {
                    float ehDist = sqrt(pow((*it).pos.X() - photon_trajectory[(*it).layer].first, 2)
                                   + pow((*it).pos.Y() - photon_trajectory[(*it).layer].second, 2));

                    if(ehDist < 8.7 && (*it).layer < firstNearPhLayer) {
                        firstNearPhLayer = (*it).layer;
                    }
                }
            }

            // Find straight MIP tracks:

            std::sort(trackingHitList.begin(), trackingHitList.end(), [](HitData ha, HitData hb) {return ha.layer > hb.layer;});
  
            float cellWidth = 8.7;

            for(int iHit = 0; iHit < trackingHitList.size(); iHit++) {
                int track[34]; // List of hit numbers in track (34 = Maximum theoretical length)
                int currenthit;
                int trackLen;

                track[0] = iHit;
                currenthit = iHit;
                trackLen = 1;

                // Search for hits to add to the track: If hit is in the next two layers behind the current hit, consider adding

                for(int jHit = 0; jHit < trackingHitList.size(); jHit++) {
                    if(trackingHitList[jHit].layer == trackingHitList[currenthit].layer || trackingHitList[jHit].layer 
                    > trackingHitList[currenthit].layer + 3) {
                        continue;  // If not in the next two layers, continue
                    }

                    // If it is: Add to track if new hit is directly behind the current hit
                    if(trackingHitList[jHit].pos.X() == trackingHitList[currenthit].pos.X() && trackingHitList[jHit].pos.Y()
                    == trackingHitList[currenthit].pos.Y()) {
                        track[trackLen] = jHit;
                        trackLen++;
                    }
                }

                // Confirm that the track is valid
                if(trackLen < 2) continue; // Track must contain at least 2 hits
                float closest_e = distTwoLines(trackingHitList[track[0]].pos, trackingHitList[track[trackLen - 1]].pos, e_traj_start, e_traj_end);
                float closest_p = distTwoLines(trackingHitList[track[0]].pos, trackingHitList[track[trackLen - 1]].pos, p_traj_start, p_traj_end);

                // Make sure that the track is near the photon trajectory and away from the electron trajectory
                // Details of these constraints may be revised
                if(closest_p > cellWidth and closest_e < 2*cellWidth) continue;
                if(trackLen < 4 and closest_e > closest_p) continue;

                // If track found, increment nStraightTracks and remove all hits in track from future consideration
                if(trackLen >= 2) {
                    for(int kHit = 0; kHit < trackLen; kHit++) {
                        trackingHitList.erase(trackingHitList.begin() + track[kHit]);
                    }

                    // The *current* hit will have been removed, so iHit is currently pointing to the next hit
                    iHit--; // Decrement iHit so no hits will get skipped by iHit++
                    nStraightTracks++;
                }

                // Optional addition: Merge nearby straight tracks. Not necessary for veto

            }

            // Linreg tracking:

            for(int iHit = 0; iHit < trackingHitList.size(); iHit++) {
                int track[34];
                int trackLen;
                int currenthit;
                int hitsInRegion[50]; // Hits being considered at one time
                int nHitsInRegion; // Number of hits under consideration
                TMatrixD svdMatrix(3, 3);
                TMatrixD Vm(3, 3);
                TMatrixD hdt(3, 3);
                TVector3 slopeVec;
                TVector3 hmean;
                TVector3 hpoint;
                float r_corr_best;
                int hitNums_best[3]; // Hit numbers of current best track candidate
                int hitNums[3];

                trackLen = 0;
                nHitsInRegion = 1;
                currenthit = iHit;
                hitsInRegion[0] = iHit;

                // Find all hits within 2 cells of the primary hit
                for(int jHit = 0; jHit < trackingHitList.size(); jHit++) {
                    float dstToHit = (trackingHitList[iHit].pos - trackingHitList[jHit].pos).Mag();
                    if(dstToHit <= 2*cellWidth) {
                        hitsInRegion[nHitsInRegion] = jHit;
                        nHitsInRegion++;
                    }
                }

                // Look at combinations of hits within the region (Do not consider the same combination twice)
                hitNums[0] = iHit;
                for(int jHit = 1; jHit < nHitsInRegion - 1; jHit++) {
                    hitNums[1] = jHit;
                    for(int kHit = jHit + 1; kHit < nHitsInRegion; kHit++) {
                        hitNums[2] = kHit;
                        for(int hInd = 0; hInd < 3; hInd++) {

                        // hmean = Geometric mean, subtract off from hits to improve SVD performance
                        hmean(hInd) = (trackingHitList[hitNums[0]].pos(hInd) +
                                      trackingHitList[hitNums[1]].pos(hInd) +
                                      trackingHitList[hitNums[2]].pos(hInd))/3.0;
                        }

                        for(int hInd = 0; hInd < 3; hInd++) {
                            for(int lInd = 0; lInd < 3; lInd++) {
                                hdt(hInd, lInd) = trackingHitList[hitNums[hInd]].pos(lInd) - hmean(lInd);
                            }
                        }

                        // Perform "linreg" on selected points
                        TDecompSVD svdObj = TDecompSVD(hdt);
                        bool decomposed = svdObj.Decompose();
                        if(!decomposed) continue;

                        Vm = svdObj.GetV(); // First col of V matrix is the slope of the best-fit line
                        for (int hInd = 0; hInd < 3; hInd++) {
                            slopeVec(hInd) = Vm[0][hInd];
                        }

                        hpoint = slopeVec + hmean; // hmean, hpoint are points on the best-fit line

                        // Linreg complete: Now have best-fit line for 3 hits under consideration
                        // Check whether the track is valid: r^2 must be high, and the track must plausibly originate from the photon
                        float closest_e = distTwoLines(hmean, hpoint, e_traj_start, e_traj_end);
                        float closest_p = distTwoLines(hmean, hpoint, p_traj_start, p_traj_end);

                        // Projected track must be close to the photon; details may change after future study
                        if(closest_p > cellWidth or closest_e < 1.5*cellWidth) continue;

                        // Find r^2
                        float vrnc = (trackingHitList[hitNums[0]].pos - hmean).Mag() +
                                     (trackingHitList[hitNums[1]].pos - hmean).Mag() +
                                     (trackingHitList[hitNums[2]].pos - hmean).Mag(); // ~Variance
                        float sumerr = distPtToLine(trackingHitList[hitNums[0]].pos, hmean, hpoint) +
                                       distPtToLine(trackingHitList[hitNums[1]].pos, hmean, hpoint) +
                                       distPtToLine(trackingHitList[hitNums[2]].pos, hmean, hpoint); // Sum of |errors|
                        float r_corr = 1 - sumerr/vrnc;

                        // Check whether r^2 exceeds a low minimum r_corr: "Fake" tracks are still much more common in background, so making the algorithm
                        // oversensitive doesn't lower performance significantly
                        if(r_corr > r_corr_best and r_corr > .6) {
                            r_corr_best = r_corr;
                            trackLen = 0;
                            for(int k = 0; k < 3; k++) { // Only looking for 3-hit tracks currently
                                track[k] = hitNums[k];
                                trackLen++;
                            }
                        }
                    }
                }

                // Ordinarily, additional hits in line with the track would be added here. However, this doesn't affect the results of the simple veto
                // Exclude all hits in a found track from further consideration
                if(trackLen >= 2) {
                    nLinregTracks++;
                    for(int kHit = 0; kHit < trackLen; kHit++) {
                        trackingHitList.erase(trackingHitList.begin() + track[kHit]);
                    }
                    iHit--;
                }
            }

            ntuple_.setVar<int>("nStraightTracks", nStraightTracks);
            ntuple_.setVar<int>("nLinregTracks", nLinregTracks);
            ntuple_.setVar<int>("firstNearPhLayer", firstNearPhLayer);
            ntuple_.setVar<float>("epAng", epAng);
            ntuple_.setVar<float>("epSep", epSep);

            // Reset hex readout variable for next event
            hexReadout_ = 0;

        }

        int nNoiseHits = 0;
        float noiseEnergy = 0.0;
        float trigEnergy = 0.0;

        int nRecHits = 0;
        int nSimHits = 0;

        // Get the SimHit collection for this event
        if(event.exists(ecalSimHitCollectionName_)) {
            auto ecalSimHits{event.getCollection<SimCalorimeterHit>(ecalSimHitCollectionName_)};
    
            std::sort(ecalSimHits.begin(), ecalSimHits.end(), [](const SimCalorimeterHit &lhs, const SimCalorimeterHit &rhs) {
                return lhs.getID() < rhs.getID();
            }
            );

            for(auto simHit : ecalSimHits) {
                if(simHit.getEdep() > 0) nSimHits++;
                simHitXv.push_back(simHit.getPosition()[0]);
                simHitYv.push_back(simHit.getPosition()[1]);
                simHitZv.push_back(simHit.getPosition()[2]);
                detID_ = EcalID(simHit.getID());
                int layer = detID_.getLayerID();
                simHitLayerv.push_back(layer);
                simHitEnergyv.push_back(simHit.getEdep());
            }
        }

        if(event.exists(ecalRecHitCollectionName_)) {
            auto ecalRecoHits{event.getCollection<EcalHit>(ecalRecHitCollectionName_)};
            std::sort(ecalRecoHits.begin(), ecalRecoHits.end(), [](const EcalHit &lhs, const EcalHit &rhs) {
                return lhs.getID() < rhs.getID();
            }
            );
    
            for(auto recHit : ecalRecoHits) {
                if(recHit.getEnergy() > 0)
                    nRecHits++;
                if(recHit.isNoise()) {
                    nNoiseHits++;
                    noiseEnergy += recHit.getEnergy();
                }
                int rawID = recHit.getID();
                double x, y, z;
                ecalHexReadout.getCellAbsolutePosition(rawID, x, y, z);
                detID_ = EcalID(rawID);
                int layer = detID_.getLayerID();
                if(layer < 20) {
                    trigEnergy += recHit.getEnergy();
                }
                hitXv.push_back(x);
                hitYv.push_back(y);
                hitZv.push_back(z);
                hitLayerv.push_back(layer);
                recHitEnergyv.push_back(recHit.getEnergy());
                recHitAmplitudev.push_back(recHit.getAmplitude());
                float totalSimEDep = 0.0;
                int nSimHitMatch = 0;
                if(event.exists(ecalSimHitCollectionName_)) {
                    auto ecalSimHits{event.getCollection<SimCalorimeterHit>(ecalSimHitCollectionName_)};
                    std::sort(ecalSimHits.begin(), ecalSimHits.end(), [](const SimCalorimeterHit &lhs, const SimCalorimeterHit &rhs) {
                        return lhs.getID() < rhs.getID();
                    }
                    );
                    for(auto simHit : ecalSimHits ) {
                        if(simHit.getID() == rawID) {
                            totalSimEDep += simHit.getEdep();
                            nSimHitMatch++;
                        }
                        else if(simHit.getID() > rawID) {
                            break;
                        }
                    }
                    simHitMatchEnergyv.push_back(totalSimEDep);

                    // If isNoise flag isn't set check for absence of simhit match to identify noise hits
                    if(!recHit.isNoise() && nSimHitMatch == 0) {
                        nNoiseHits++;
                        noiseEnergy += recHit.getEnergy();
                    }
                }
            }
        }

        int nElectrons = 0;
        if(event.exists(simParticleCollectionName_)) {
            auto particleMap{event.getMap<int, SimParticle>(simParticleCollectionName_)};
            for(auto const& [trackID, particle] : particleMap) {
                if(particle.getPdgID() == 11 && particle.getParents().size() == 0 && particle.getProcessType() == 0) nElectrons++;
            }
        }

        ntuple_.setVar<int>("nNoiseHits", nNoiseHits);  
        ntuple_.setVar<float>("noiseEnergy", noiseEnergy);  
        ntuple_.setVar<float>("trigEnergy", trigEnergy);  
        ntuple_.setVar<int>("nRecHits", nRecHits);
        ntuple_.setVar<int>("nSimHits", nSimHits);
        ntuple_.setVar<int>("nElectrons", nElectrons);

        hitTree_->Fill();
        simHitTree_->Fill();

    }

    /* Calculate where trajectory intersects ECAL layers using position and momentum at scoring plane */
    std::vector<std::pair<float, float>> ECalVetoAnalyzer::getTrajectory(std::vector<double> momentum, std::vector<float> position) {
        std::vector<XYCoords> positions;
        for (int iLayer = 0; iLayer < nECalLayers; iLayer++) {
            float posX = position[0] + (momentum[0]/momentum[2])*(hexReadout_->getZPosition(iLayer) - position[2]);
            float posY = position[1] + (momentum[1]/momentum[2])*(hexReadout_->getZPosition(iLayer) - position[2]);
            positions.push_back(std::make_pair(posX, posY));
        }
        return positions;
    }

    // MIP tracking functions:

    float ECalVetoAnalyzer::distTwoLines(TVector3 v1, TVector3 v2, TVector3 w1, TVector3 w2) {
        TVector3 e1 = v1 - v2;
        TVector3 e2 = w1 - w2;
        TVector3 crs = e1.Cross(e2);
        if (crs.Mag() == 0) {
            return 100.0; // Arbitrary large number; edge case that shouldn't cause problems
        }
        else {
            return std::abs(crs.Dot(v1 - w1)/crs.Mag());
        }
    }

    float ECalVetoAnalyzer::distPtToLine(TVector3 h1, TVector3 p1, TVector3 p2) {
        return ((h1 - p1).Cross(h1 - p2)).Mag()/(p1 - p2).Mag();
    }

} //

DECLARE_ANALYZER_NS(ldmx, ECalVetoAnalyzer)
