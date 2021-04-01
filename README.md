# Python Implementation of BDT

## Purpose
Faster development. Eliminates need for ldmx-analysis or other dependencies.

## Requirements
- Working installation of ldmx-sw v3.0.0-alpha or later

## Running Examples

**Train**
```
ldmx python3 train.py -b <bkgd_file> -s <signal_file> -o dummy --max_evt 100
```
This will produce a file `dummy_weights.pkl` that stores the weights of the trained BDT.

**Evaluate**
```
ldmx python3 eval.py --bdt <bdt.pkl-to-use> --max_evt 100 <file-to-evaluate>
```
This will produce a file `bdt_1_<file-to-evalulate>` that is identical to `<file-to-evaluate>` except
that the `discValue_` member of `EcalVeto` has been updated with the value calculated by the chosen BDT.

## EaT Notes

- Using Trigger-Skimmed 1e12 EoT Equivalent Enriched Nuclear Background Sample as training set
  (1e13 EoT Equivalent set will be testing set)
- Using three 20k event files from each mass point as training set (after trigger skim)
  (the rest -- 47 20k event files for each mass point -- will be testing set)
