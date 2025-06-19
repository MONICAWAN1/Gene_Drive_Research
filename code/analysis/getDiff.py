import sys, os, math
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import haploid, haploid_se, wm
from utils import load_pickle, save_pickle, euclidean

def get_eq(params):
    s, c, h = params['config']
    sn = 0.5*(1-c)*(1-h*s)
    if params['conversion'] == 'zygotic':
        sc = c*(1-s)
    else:
        sc = c*(1-h*s)

    eqs = {'q1':0, 'q2':1}

    if 4*sn+2*sc+s-2 != 0:
        q3 = (2*sn+2*sc-1)/(4*sn+2*sc+s-2)
    else:
        q3 = 'NA'
        print(f"no eq result, config = {params['config']}")
    eqs['q3'] = q3
    return eqs

def getdiff(currH, mapfunction, gdFile):
    '''
    given gd results and one mapping result
    saves a dictionary (s,c,h): error of map
    store results as .txt in mapping_diff_txt folder and as .pickle in mapping_diff folder
    '''
    # test s, c values
    ts = 0.2
    tc = 0.9
    state = "fix"

    loadFile = f"gd_simulation_results/h{currH}_allgdres{gdFile}G.pickle"
    fileName = f"mapping_diff_txt/h{currH}_mappingdiff_{mapfunction}{gdFile}_G_{state}.txt"
    f_out = open(fileName, 'w')
    print(f"writing to {fileName}")
    f_out.write(f"Gene Drive Configuration\t\t{mapfunction} result\t\tMSE error compared to Gene Drive\n")

    # load gd curves and stability results
    gd_results = load_pickle(loadFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = load_pickle(f"h{currH}_gametic_stability_res.pickle")

    savedPickle = f"mapping_diff/h{currH}_mappingdiff_{mapfunction}{gdFile}_{state}.pickle"

    # with open('pickle/allngdres.pickle', 'rb') as f1:
    #     wm_results = pickle.load(f1)
    gridResFile = f"mapping_result/h{currH}_{mapfunction}{gdFile}_G_{state}.pickle"
    gridResult = load_pickle(gridResFile)

    sMap_grid, wms_grid = gridResult['map'], gridResult['ngC']
    # sMap_grad, wms_grad = gradResult['map'], gradResult['ngC'] ## comment out for switching mapping method
    print('len of grid map', len(sMap_grid))

    diffmap = dict()
    ngd_config = []

    for (s, c, h) in gd_configs:
        # if math.isclose(s, 0.2) and math.isclose(c, 0.9):
        # if ((s, c, h) in sMap_grid and ((s, c, h), 1.0) in stabilityRes['Fixation']) or (math.isclose(h, 0.5) and gd_res[(s, c, h)]['state']=='fix'):
        params_eq = {'config': (s, c, h), 'conversion': 'gametic'}
        eq = get_eq(params_eq)['q3']

        if ((s, c, h) in sMap_grid and ((s, c, h), 1.0) in stabilityRes['Fixation']):
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
    getdiff(0.3, "grid", "001")

if __name__ == '__main__':
    main()