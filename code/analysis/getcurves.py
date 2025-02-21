import numpy as np
import math
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import run_model, wm, haploid
from utils import save_pickle

### run simulation for a list of configurations, returns a list of configs and a dictionary {config: curve} #############
def gd_simulation(params):
    '''
    decides what configurations the current simulation set includes
    '''
    minVal, maxVal = 0, 1.0 # gd params all in range [0, 1]
    configs = []
    s_vals = np.arange(minVal, maxVal, 0.1)
    c_vals = np.arange(0, maxVal, 0.1)
    h_vals = np.arange(0, maxVal, 0.1)

    # get a list of configurations for simulations 
    for s in s_vals:
        if s>0:
            for c in c_vals:
                if c>0:
                    for h in h_vals:
                        if math.isclose(h, params['h']):
                            configs.append((round(float(s), 3), round(float(c), 3), round(float(h), 3)))

    gd_results = dict()
    for (s, c, h) in configs:
        # print('gd:', (s, c, h))
        params['s'], params['c'], params['h'] = s, c, h
        gd_res = run_model(params)
        gd_results[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = gd_res
    
    return configs, gd_results

### pickle dump ngd results: {(s,h): curve} and gd results [configlist, gd_res]
def getcurves(params):
    '''
    pickle dump ngd results: {(s,h): curve} and gd results [configlist, gd_res] 
    where gd_res: (s,c,h): {'q':, 'state':}
    CHANGE SAVED PICKLE FILE NAME!
    '''
    minVal, maxVal, step = -30, 1, 0.01
    # s_range = np.arange(minVal, maxVal, step)
    # h_range = np.arange(0, maxVal, step)
    # wm_results = dict()
    # print(s_range, h_range)
    # for s_nat in s_range:  
    #     for h_nat in h_range: 
    #         wm_curve = wm(s_nat, h_nat, params['target_steps'], params['q0'])['q']
    #         wm_results[(round(s_nat, 3), round(h_nat, 3))] = wm_curve
    gd_configs, gd_res = gd_simulation(params)
    gd_results = [gd_configs, gd_res]  # gd_configs: list of tuples, gd_res: map {config: curve}
    print('length of configs', len(gd_configs))
    # with open('pickle/allngdres.pickle', 'wb') as f1:
    #     pickle.dump(wm_results, f1)
    save_pickle(f"h{params['h']}_allgdresG.pickle", gd_results)  ### allgdres001: step = 0.01

def gethaploid(params):
    '''
    run haploid model to get all result in the param space
    '''
    hapRes = dict()
    minVal, maxVal, step = -30, 1, 0.01
    s_range = np.arange(minVal, maxVal, step)
    for s in s_range:
        params['s'] = s
        hapC = haploid(params)
        hapRes[round(s, 3)] = hapC

    save_pickle(f"allhapresG.pickle", hapRes)

colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']

def main():
    params = {'n': 500, 'h': 0.5, 'target_steps': 40000, 'q0': 0.001}
    getcurves(params)
    # gethaploid(params)

if __name__ == '__main__':
    main()