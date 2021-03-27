# Python Implimentation of BDT

Purpose: Faster development and eliminates need for ldmx-analysis and other dependencies\
Requirments: Working install of `ldmx-sw-v2.3.0` or greater and `v12` samples.\
       +     Only tested with container including numpy, xgboost, sklearn, and matplotlib packages.
             
Currently set to to run seg-mip BDT.

Example TreeMaker command to make flat trees from event samples:
```
ldmx python3 treeMaker.py --interactive -i <absolute_path_to_input> -g <label_for_input_eg_PN> --out <absolute_output_path> -m <max_events>
```
`--indirs` can be used to use all files from given directories. More information can be found in `mods/ROOTmanager.py`

Example bdtMaker command to train BDT: \
`Coming soon`

Example bdtEval command to evaluate trained BDT on test samples: \
`coming soon`
