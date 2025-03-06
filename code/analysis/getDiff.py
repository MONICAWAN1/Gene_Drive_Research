import sys, os, math
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import haploid, haploid_se, wm
from utils import load_pickle, save_pickle, euclidean

def getdiff(currH, mapfunction, gdFile):
    '''
    given gd results and one mapping result
    saves a dictionary (s,c,h): error of map
    '''
    loadFile = f"h{currH}_allgdres{gdFile}G.pickle"
    fileName = f"s0.4_c0.4_h{currH}_mappingdiff_{mapfunction}{gdFile}_G_fix.txt"
    f_out = open(fileName, 'w')
    print(f"writing to {fileName}")
    f_out.write(f"Gene Drive Configuration\t\t{mapfunction} result\t\tMSE error compared to Gene Drive\n")

    # load gd curves and stability results
    gd_results = load_pickle(loadFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = load_pickle(f"h{currH}_gametic_stability_res.pickle")

    savedPickle = f"s0.4_c0.4_h{currH}_mappingdiff_{mapfunction}{gdFile}.pickle"

    # with open('pickle/allngdres.pickle', 'rb') as f1:
    #     wm_results = pickle.load(f1)
    print(f"h{currH}_{mapfunction}{gdFile}_G_fix.pickle")
    gridResult = load_pickle(f"s0.4_c0.4_h{currH}_{mapfunction}{gdFile}_G_fix.pickle")

    sMap_grid, wms_grid = gridResult['map'], gridResult['ngC']
    # sMap_grad, wms_grad = gradResult['map'], gradResult['ngC'] ## comment out for switching mapping method
    print('len of grid map', len(sMap_grid))

    diffmap = dict()
    ngd_config = []
    s = 0.6
    c = 0.4
    h = currH

    for (s, c, h) in gd_configs:
        if math.isclose(s, 0.4) and math.isclose(c, 0.6):
        # if ((s, c, h) in sMap_grid and ((s, c, h), 1.0) in stabilityRes['Fixation']) or (math.isclose(h, 0.5) and gd_res[(s, c, h)]['state']=='fix'):
            gd_curve = gd_res[(s, c, h)]['q']
            if 'hap' in mapfunction:
                ngd_s = sMap_grid[(s, c, h)]
                params = {'s': ngd_s, 'target_steps': 40000, 'q0': 0.001} # set parameters to the mapped s value for ngd model
                ngd_curve = haploid(params)['q']
            else:
                ngd_s, ngd_h = sMap_grid[(s, c, h)][0], sMap_grid[(s, c, h)][1]
                ngd_curve = wm(ngd_s, ngd_h, 40000, 0.001)['q']
            

            # paramSe = {'s':s, 'c':c, 'n': 500, 'h':h, 'target_steps': 40000, 'q0': 0.001}
            # ngd_curve = haploid_se(paramSe)['q']

            diff = euclidean(ngd_curve, gd_curve)
            diffmap[(s, c, h)] = diff
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\tmapped to {sMap_grid[(s, c, h)]}\t\tMSE error={'%.6f' % diff}\n")
            # ngd_config.append(ngd_key)
    save_pickle(savedPickle, diffmap)

def main():
    getdiff(0.0, "gradient")

if __name__ == '__main__':
    main()