import numpy as np
import math
import pickle
from utils import euclidean
from models import wm
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_pickle
from models import haploid

def loadStability(currH):
    return load_pickle(f"h{currH}_gametic_stability_res.pickle")

def loadGres(currH):
    '''
    remember to change filename in getDiff before run map
    '''
    return load_pickle(f"h{currH}_allgdresG.pickle")

def f(ngd_s, ngd_h, s, c, h, params):
    gd_results = loadGres(h)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    gd_curve = gd_res[(s, c, h)]['q']
    f = 0
    ngd_curve = wm(ngd_s, ngd_h, params['target_steps'], params['q0'])['q']

    return euclidean(ngd_curve, gd_curve)

def optimization(ngd_s, s, c, gd_h, params):
    N = 0
    delta = 0.001
    curr_h = gd_h + delta
    new_h = curr_h

    # Newton-Raphson optimization based on f(h)
    while N < 1000:
        print(curr_h)
        f_h2 = f(ngd_s, curr_h+delta, s, c, gd_h, params)
        f_h1 = f(ngd_s, curr_h-delta, s, c, gd_h, params)
        f_h = f(ngd_s, curr_h, s, c, gd_h, params)
        print('f values', f_h, f_h1, f_h2)
        df = (f_h2 - f_h1) / (2*delta)
        print(df)
        if df == 0:
            break
        new_h = curr_h - f_h/df
        if f(ngd_s, new_h, s, c, gd_h, params) > f(ngd_s, curr_h, s, c, gd_h, params):
            break
        curr_h = new_h
        N += 1
    
    return curr_h

def optimization_s(ngd_s, s, c, gd_h, params):
    N = 0
    delta = 0.001
    curr_s = ngd_s + delta
    print('init', curr_s)
    new_s = curr_s

    # Newton-Raphson optimization based on f(h)
    while N < 10:
        curr_s = new_s
        print(curr_s)
        f_h2 = f(curr_s+delta, gd_h,  s, c, gd_h, params)
        f_h1 = f(curr_s-delta, gd_h, s, c, gd_h, params)
        f_h = f(curr_s, gd_h, s, c, gd_h, params)
        print('s values', f_h, f_h1, f_h2)
        df = (f_h2 - f_h1) / (2*delta)
        if df == 0:
            break
        if f(new_s, gd_h, s, c, gd_h, params) > f(curr_s, gd_h, s, c, gd_h, params):
            break
        new_s = curr_s - f_h/df
        N += 1
    
    return curr_s

def opt_r(F, grad_F, vars, s, c, h, params, alpha=1e-4, beta=0.9, r_init=1.0):
    r = r_init
    while True:
        new_vars = vars - r * grad_F
        new_F = f(new_vars[0], new_vars[1], s, c, h, params)
        if  new_F <= F - alpha * r * np.dot(grad_F, grad_F):
            return r
        r *= beta

### gradient haploid helper functions ##########################################################
def f_hap(ngd_s, s, c, h, params):
    gd_results = loadGres(h)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    gd_curve = gd_res[(s, c, h)]['q']

    # haploid mapping
    params['s'] = ngd_s
    ngd_curve = haploid(params)['q']
    f = 0
    return euclidean(ngd_curve, gd_curve)

def opt_r_hap(F, grad_F, vars, s, c, h, params, alpha=1e-4, beta=0.0099, r_init=0.1):
    r = r_init
    while True:
        new_vars = vars - r * grad_F
        new_F = f_hap(new_vars[0], s, c, h, params)
        if  new_F <= F - alpha * r * np.dot(grad_F, grad_F):
            return r
        r *= beta

### gradient haploid & diploid functions ##########################################################

def gradient(ngd_s, ngd_h, s, c, h, params):
    vars = np.array([ngd_s, ngd_h])
    delta = 0.001
    iterations = 500
    grad_F = np.zeros(len(vars))
    F = 0
    prev_F = F+10

    for i in range(iterations):
        print('iteration', i, s, c, h)
        F = f(vars[0], vars[1], s, c, h, params)
        if abs(F - prev_F) < 1e-10:
            print("Stopping: function change is too small.")
            break
        for var_i in range(len(vars)):
            vars_plus_delta = vars.copy()
            vars_plus_delta[var_i] += delta
            f_s1 = f(vars_plus_delta[0], vars_plus_delta[1], s, c, h, params)
            df_s = (f_s1 - F)/delta
            grad_F[var_i] = df_s

        
        # find step size minimalize f((s, h) - r*grad(s, h))
        r = opt_r(F, grad_F, vars, s, c, h, params)
        print(r)
        new_vars = vars - r*grad_F
        print('1st', F, new_vars)

        # if new_vars[1] < 0 or new_vars[1] > 1:
        #     print(vars)
        #     break
        prev_F = F

        new_vars[0] = np.clip(new_vars[0], -100, 1) 
        new_vars[1] = np.clip(new_vars[1], 0, 1)  # Constraint on 'h'
        # if math.isclose(f(new_vars[0], new_vars[1], s, c, h, params), F, rel_tol=1e-5):
        #     break
        vars = new_vars
        print('2nd', F, vars)
    return vars

def hap_gradient(ngd_s, s, c, h, params):
    vars = np.array([ngd_s])
    delta = 0.001
    iterations = 500
    grad_F = np.zeros(len(vars))
    F = 0
    prev_F = F+10

    for i in range(iterations):
        print('iteration', i, s, c, h)
        F = f_hap(vars[0], s, c, h, params)
        if abs(F - prev_F) < 1e-8:
            print("Stopping: function change is too small.")
            break
        for var_i in range(len(vars)):
            vars_plus_delta = vars.copy()
            vars_plus_delta[var_i] += delta
            f_s1 = f_hap(vars_plus_delta[0], s, c, h, params)
            df_s = (f_s1 - F)/delta
            grad_F[var_i] = df_s

        
        # find step size minimalize f((s, h) - r*grad(s, h))
        r = opt_r_hap(F, grad_F, vars, s, c, h, params)
        print(r)
        new_vars = vars - r*grad_F
        print('1st', F, new_vars)

        # if new_vars[1] < 0 or new_vars[1] > 1:
        #     print(vars)
        #     break
        prev_F = F

        new_vars[0] = np.clip(new_vars[0], -100, 1) 
        # if math.isclose(f(new_vars[0], new_vars[1], s, c, h, params), F, rel_tol=1e-5):
        #     break
        vars = new_vars
        print('2nd', F, vars)
    return vars

def hap_mapping(params):
    seffMap = dict()
    hapMappedCurves = []
    f_out = open(f"h{params['h']}_hap_grid_G_fix_map.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\ts in haploid ngd population\n")

    gd_results = loadGres(params['h'])
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = loadStability(params['h'])
    # print('params', gd_configs)

    hap_results = load_pickle("allhapresG.pickle")

    for (s, c, h) in gd_configs:
        # only testing for s = 0.4, c=0.6
        # if math.isclose(s, 0.4) and math.isclose(c, 0.6):
        # if gd_res[(s, c, h)]['state'] == 'fix':
        if ((s, c, h), 1.0) in stabilityRes['Fixation']:
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s = None
            best_h = h
        
            for ngd_key, ngd_curve in hap_results.items():
                diff = euclidean(ngd_curve['q'], gd_curve)
                if diff < best_diff:
                    best_diff = diff
                    best_s = ngd_key

            print('before gradient:', best_s)

            ### haploid
            hap_params = {'s': best_s, 'q0': params['q0'], 'target_steps': params['target_steps']}
            best_curve = haploid(hap_params)['q']
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = round(best_s, 3)
            hapMappedCurves.append(best_curve)
    return {'map': seffMap, 'ngC': hapMappedCurves}

def hap_gradient_mapping(params):
    seffMap = dict()
    hapMappedCurves = []
    f_out = open(f"h{params['h']}_hap_gradient_G_fix_map.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\ts in haploid ngd population\n")

    gd_results = loadGres(params['h'])
    gd_configs, gd_res = gd_results[0], gd_results[1]
    # print('params', gd_configs)
    stabilityRes = loadStability(params['h'])
    grid_results = load_pickle(f"h{params['h']}_hap_grid_G_fix.pickle")

    for (s, c, h) in gd_configs:
        # only testing for s = 0.4, c=0.6
        # if math.isclose(s, 0.8) and math.isclose(c, 0.9):
        if ((s, c, h), 1.0) in stabilityRes['Fixation']:
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s = grid_results['map'][(s, c, h)]

        # gradient descent or newton_raphson
            best_vars = hap_gradient(best_s, s, c, h, params)
            best_s = best_vars[0]

            ### haploid
            hap_params = {'s': best_s, 'q0': params['q0'], 'target_steps': params['target_steps']}
            best_curve = haploid(hap_params)['q']
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = round(best_s, 3)
            hapMappedCurves.append(best_curve)
    return {'map': seffMap, 'ngC': hapMappedCurves}

#### grid search only ################
def grid_mapping(params):
    seffMap = dict()
    ngd_mapped = []
    f_out = open('outputs/grid_G_h0_s02c03.txt', 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h) in NGD population (S_Effective)\n")

    gd_results = loadGres(params['h'])
    gd_configs, gd_res = gd_results[0], gd_results[1]

    ngd_results = load_pickle("allngdres.pickle")

    for (s, c, h) in gd_configs:
        # if gd_res[(s, c, h)]['state'] == 'fix':
        if s < c:
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_ngd_config = None
            best_ngd_curve = None
            for ngd_key, ngd_curve in ngd_results.items():
                diff = euclidean(ngd_curve, gd_curve)
                if diff < best_diff:
                    best_diff = diff
                    best_ngd_config = ngd_key
                    best_ngd_curve = ngd_curve

            seffMap[(s, c, h)] = best_ngd_config
            ngd_mapped.append(best_ngd_curve)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_ngd_config[0]}, h={'%.3f' % best_ngd_config[1]}\terror={'%.3f' % best_diff}\n")
    
    # print(seffMap)
    
    return {'map': seffMap, 'ngC': ngd_mapped}
        


#### grid + gradient ########################
def gradient_mapping(params):
    seffMap = dict()
    wms = []
    f_out = open(f"h{params['h']}_grad_G_fix.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h)in NGD population\n")

    gd_results = loadGres(params['h'])
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = loadStability(params['h'])
    grid_results = load_pickle(f"h{params['h']}_hap_grid_G_fix.pickle")

    for (s, c, h) in gd_configs:
        # if math.isclose(s, 0.8) and math.isclose(c, 0.9):
        if ((s, c, h), 1.0) in stabilityRes['Fixation']:
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s = grid_results['map'][(s,c,h)]
            print('before gradient', best_s)
            best_h = h
        
            # for ngd_key, ngd_curve in wm_results.items():
            #     diff = euclidean(ngd_curve, gd_curve)
            #     if diff < best_diff:
            #         best_diff = diff
            #         best_config = ngd_key
                    
        
        # gradient descent or newton_raphson

            best_vars = gradient(best_s, best_h, s, c, h, params)
            best_s, best_h = best_vars[0], best_vars[1]
            # best_s = optimization_s(best_s, s, c, h, params)
            # best_h = optimization(best_s, s, c, h, params)

            best_curve = wm(best_s, best_h, params['target_steps'], params['q0'])['q']
            best_diff = euclidean(best_curve, gd_curve)
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}, h={'%.3f' % best_h}\terror={'%.3f' % best_diff}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = (round(best_s, 3), round(float(best_h), 3))
            wms.append(best_curve)
    return {'map': seffMap, 'ngC': wms}