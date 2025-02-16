import sys, os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import haploid
from utils import load_pickle, save_pickle, euclidean

def getdiff():
    gd_results = load_pickle("h05_allgdres001G.pickle")
    gd_configs, gd_res = gd_results[0], gd_results[1]

    # with open('pickle/allngdres.pickle', 'rb') as f1:
    #     wm_results = pickle.load(f1)
    gridResult = load_pickle("h05_hap_grid_gametic_fix.pickle")

    sMap_grid, wms_grid = gridResult['map'], gridResult['ngC']
    # sMap_grad, wms_grad = gradResult['map'], gradResult['ngC']

    diffmap = dict()
    ngd_config = []
    s = 0.6
    c = 0.4
    h = 0.5

    for (s, c, h) in gd_configs:
        if gd_res[(s, c, h)]['state'] == 'fix':
            gd_curve = gd_res[(s, c, h)]['q']
            ngd_s = sMap_grid[(s, c, h)]
            params = {'s': ngd_s, 'target_steps': 40000, 'q0': 0.001}
            ngd_curve = haploid(params)['q']
            # print(ngd_s, ngd_curve)
            # print('found', gd_curve)

            diff = euclidean(ngd_curve, gd_curve)/min(len(ngd_curve), len(gd_curve))
            diffmap[(s, c, h)] = diff
            # ngd_config.append(ngd_key)
    print(diffmap)
        
    save_pickle("h05_mappingdiff_grid_hap.pickle", diffmap)

def main():
    getdiff()

if __name__ == '__main__':
    main()