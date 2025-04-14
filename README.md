# Gene Drive project

Repository for simulation models

## Top-Level Directories

### [`analysis/`](/analysis)
Files that get simulation results, run analysis, and generate result figures

### [`models/`](/models)
Gene drive and non-gene-drive models

### [`scripts/`](/scripts)
Scripts to run files in analysis for mapping or plotting results

### [`utils/`](/utils)
Helper functions

Mapping Result Analysis Pipeline:
1. run get_curves.py:
    - modify h in params
    - select type of model to run and save
    - adjust step (0.01/0.1)
2. stability analysis: run stability.py
    - modify s,c,h in main()
    - (optional) change s and c range in runall(params)
3. get mapping: 
    - run run_mapping.py <map_function> <h> <gdFile>
        - gdFile example: h{currH}_allgdRes001G
        - change stability regime condition in the for loop
        - CHANGE FILENAME in run_mapping
    - Change file name and loop condition in getDiff.py
4. plot mapping: 
    -   run run_plot.py <h>


General Pipeline for Mapping:
Goal: Compare mapping errors 
1. Fix/Loss regimes:
     - haploid_se vs haploid grid search/gradient
     - diploid grid search

2. Stable/Unstable regimes:
    - diploid grid search
        * h range can be > 1
