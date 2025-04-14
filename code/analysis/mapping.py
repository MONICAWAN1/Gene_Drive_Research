import numpy as np
import math
import pickle
from utils import euclidean
from models import wm
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sympy as sp

from .stability import derivative
from .plotting import plot_lambda_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_pickle, save_pickle
from models import haploid

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

'''
Given the NGD parameter (s, h), return the stable eq
'''
def get_eq_ngd(s, h):
    if not math.isclose(h, 0.5):
        return (h*s)/(2*h*s - s)
    return 'NA'

# def get_ngd_stability(s, h, q):
#     num = 2*q**2*(1-q)*(1-s)*(1-h*s) - q**2*(1-2*q)*(1-s)*(1-h*s) + 2*q*(1-q)**2*(2-s-h*s) + (1-2*q)*(1-q)**2*(1-h*s)

#     denom = (q**2*(1-s) + 2*(1-q)*q*(1-h*s) + (1-q)**2)**2

#     if denom != 0:
#         slope = num/denom
#     else:
#         slope = 'NA'
#         print(f"no slope result, config = {(s, h, q)}")
    
#     return slope
q, s, h = sp.symbols('q s h')

# Define the symbolic expression for q(t+1)
numerator = q**2 * (1 - s) + q * (1 - q) * (1 - h * s)
wbar = q**2 * (1 - s) + 2 * q * (1 - q) * (1 - h * s) + (1 - q)**2
q_next = numerator / wbar

# Derivative of q(t+1) with respect to q(t)
dq_next_dq = sp.simplify(sp.diff(q_next, q))

# Convert the symbolic derivative into a Python function
dq_dq_func = sp.lambdify((s, h, q), dq_next_dq, "numpy")

print(dq_dq_func)



def get_ngd_stability(s, h, q):
    try:
        slope = dq_dq_func(s, h, q)
    except ZeroDivisionError:
        slope = 'NA'
        print(f"Division by zero: config = {(s, h, q)}")
    return slope


'''
Check if current GD eq stability is the same as mapping NGD
'''
def check_state_eq(s, h, q, gdState):
    ngd_stability = get_ngd_stability(s, h, q)
    # print("++++++++++++++++++++++")
    # print(ngd_stability)
    if ngd_stability != 'NA':
        ngd_state = "Stable" if ngd_stability < 1 else "Unstable"
        return ngd_state == gdState
    return False

def loadStability(currH):
    return load_pickle(f"h{currH}_gametic_stability_res.pickle")

def loadGres(currH, gdFile):
    '''
    remember to change filename in getDiff before run map
    '''
    return load_pickle(f"gd_simulation_results/h{currH}_allgdres{gdFile}G.pickle")

def f(ngd_s, ngd_h, s, c, h, params, gdFile):
    gd_results = loadGres(h, gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    gd_curve = gd_res[(s, c, h)]['q']
    f = 0
    ngd_curve = wm(ngd_s, ngd_h, params['target_steps'], params['q0'])['q']

    # DEBUGGING PLOTS
    # if ngd_h != h:
    #     print(f"NGD_H: {ngd_h}, H: {h}")
    #     ngd_curve2 = wm(ngd_s, h, params['target_steps'], params['q0'])['q']
    #     plt.figure(figsize=(10, 8))
    #     time = np.arange(len(ngd_curve))
    #     time2 = np.arange(len(ngd_curve2))
    #     plt.plot(time, ngd_curve)
    #     plt.plot(time2, ngd_curve2)
    #     plt.xlabel("Time")
    #     plt.ylabel("Gene Drive Allele Frequency")
    #     plt.show()

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

def opt_r(F, grad_F, vars, s, c, h, params, gdFile, alpha=1e-4, beta=0.9, r_init=1.0):
    r = r_init
    while True:
        new_vars = vars - r * grad_F
        new_F = f(new_vars[0], new_vars[1], s, c, h, params, gdFile)
        if  new_F <= F - alpha * r * np.dot(grad_F, grad_F):
            return r
        r *= beta

### gradient haploid helper functions ##########################################################
def f_hap(ngd_s, s, c, h, params, gdFile):
    gd_results = loadGres(h, gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    gd_curve = gd_res[(s, c, h)]['q']

    # haploid mapping
    params['s'] = ngd_s
    ngd_curve = haploid(params)['q']
    f = 0
    return euclidean(ngd_curve, gd_curve)

def opt_r_hap(F, grad_F, vars, s, c, h, gdFile, params, alpha=1e-4, beta=0.0099, r_init=0.1):
    r = r_init
    while True:
        new_vars = vars - r * grad_F
        new_F = f_hap(new_vars[0], s, c, h, params, gdFile)
        if  new_F <= F - alpha * r * np.dot(grad_F, grad_F):
            return r
        r *= beta

### gradient haploid & diploid functions ##########################################################

def gradient(ngd_s, ngd_h, s, c, h, params, gdFile):
    vars = np.array([ngd_s, ngd_h])
    deltas = [0.001, 1]
    iterations = 1000
    grad_F = np.zeros(len(vars))
    F = 0
    prev_F = F+10
    trace = [(ngd_s, ngd_h)]
    fout = open(f"s{s}_c{c}_h{h}_gt.txt", 'w')

    for i in range(iterations):
        print('iteration', i, s, c, h)
        F = f(vars[0], vars[1], s, c, h, params, gdFile)
        print(f'ITERATION {i}')
        if abs(F - prev_F) < 1e-8:
            print("Stopping: function change is too small.")
            break
        for var_i in range(len(vars)):
            vars_plus_delta = vars.copy()
            vars_plus_delta[var_i] += deltas[var_i]
            print(f"IN LOOP with {var_i}")
            f_s1 = f(vars_plus_delta[0], vars_plus_delta[1], s, c, h, params, gdFile)
            df_s = (f_s1 - F)/deltas[var_i]
            grad_F[var_i] = df_s

        
        # find step size minimalize f((s, h) - r*grad(s, h))
        print(grad_F)
        r = opt_r(F, grad_F, vars, s, c, h, params, gdFile)
        # r = 1
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
        trace.append((vars[0], vars[1]))
        fout.write(f"Iteration {i}\nGradient: {grad_F}\tStep: {r}\tNew Vars: s={new_vars[0]}, h={new_vars[1]}\tDiff: {F}\n")
        print('2nd', F, vars)
        # break
    
    save_pickle(f"s{s}_c{c}_h{h}_gradient_trace_r.pickle", trace)
    return vars

def hap_gradient(ngd_s, s, c, h, params, gdFile):
    vars = np.array([ngd_s])
    delta = 0.001
    iterations = 500
    grad_F = np.zeros(len(vars))
    F = 0
    prev_F = F+10

    for i in range(iterations):
        print('iteration', i, s, c, h)
        F = f_hap(vars[0], s, c, h, params, gdFile)
        if abs(F - prev_F) < 1e-8:
            print("Stopping: function change is too small.")
            break
        for var_i in range(len(vars)):
            vars_plus_delta = vars.copy()
            vars_plus_delta[var_i] += delta
            f_s1 = f_hap(vars_plus_delta[0], s, c, h, params, gdFile)
            df_s = (f_s1 - F)/delta
            grad_F[var_i] = df_s

        
        # find step size minimalize f((s, h) - r*grad(s, h))
        r = opt_r_hap(F, grad_F, vars, s, c, h, params, gdFile)
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

def hap_grid_mapping(params, gdFile):
    seffMap = dict()
    hapMappedCurves = []
    f_out = open(f"h{params['h']}_hap_grid{gdFile}_G_fix_map.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\ts in haploid ngd population\n")

    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = loadStability(params['h'])
    # print('params', gd_configs)

    hap_results = load_pickle(f"allhapres0001G.pickle")

    for (s, c, h) in gd_configs:
        # only testing for s = 0.4, c=0.6
        # if math.isclose(s, 0.01) and math.isclose(c, 0.01):
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

def hap_gradient_mapping(params, gdFile):
    seffMap = dict()
    hapMappedCurves = []
    f_out = open(f"h{params['h']}_hap_gradient{gdFile}_G_fix_map.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\ts in haploid ngd population\n")

    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    # print('params', gd_configs)
    stabilityRes = loadStability(params['h'])
    grid_results = load_pickle(f"h{params['h']}_hap_grid{gdFile}_G_fix.pickle")

    for (s, c, h) in gd_configs:
        # only testing for s = 0.4, c=0.6
        # if math.isclose(s, 0.4) and math.isclose(c, 0.6):
        # if ((s, c, h), 1.0) in stabilityRes['Fixation']: 
        if gd_res[(s,c,h)]['state'] == 'fix': ### only for h = 0.5
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s = grid_results['map'][(s, c, h)]

            # gradient descent or newton_raphson
            best_vars = hap_gradient(best_s, s, c, h, params, gdFile)
            best_s = best_vars[0]

            ### haploid
            hap_params = {'s': best_s, 'q0': params['q0'], 'target_steps': params['target_steps']}
            best_curve = haploid(hap_params)['q']
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = round(best_s, 3)
            hapMappedCurves.append(best_curve)
    return {'map': seffMap, 'ngC': hapMappedCurves}

def get_delta_lambda(gd_config, ngd_config, eq):
    params = {'config':gd_config, 'currq':eq, 'conversion': "gametic"}
    gd_lambda = derivative(params)
    ngd_lambda = get_ngd_stability(ngd_config[0], ngd_config[1], eq)
    return (gd_lambda-ngd_lambda, ngd_lambda)

def find_candidate(s_ngd, h_ngd, q_ngd, state, gd_config, gd_curve, eq, s_mse_map):

    gds, gdc, gdh = gd_config
 
    best_diff = 100000
    best_ngd_config = None
    best_ngd_curve = None
    target_steps = 40000
    # best_dl = None
    # curr_eq from simulation might not be the same as the analytical eq of gd ???
    curr_eq = wm(s_ngd, h_ngd, target_steps, q_ngd)['q'][-1]
    if check_state_eq(s_ngd, h_ngd, eq, state): 
        # if s_ngd < 0:
        #     print("-----FIND MSE: S WITH THE SAME STATE:", s_ngd)
        delta_lambda, ngd_lambda = get_delta_lambda(gd_config, (s_ngd, h_ngd), eq)
        ### get MSE
        ngd_curve = wm(s_ngd, h_ngd, target_steps, q_ngd)['q']
        diff = euclidean(ngd_curve, gd_curve)
        # print(s_ngd, h_ngd, diff)
        nl = ngd_lambda if ngd_lambda != 'NA' else 0
        s_mse_map[s_ngd] = {'MSE': diff, "dl": delta_lambda, 'eq': curr_eq, "lambda": nl}
        
        # print(diff)
        # if best_diff == None or diff < best_diff:
        best_diff = diff
        best_ngd_config = (s_ngd, h_ngd)
        best_ngd_curve = ngd_curve


            
            ######  OPTION 2: If picking s_ngd based on delta_lambda: ############
            # if best_dl == None or delta_lambda < best_dl:
            #     best_dl = delta_lambda
            #     best_ngd_config = (s_ngd, h_ngd, q_ngd)
            #     best_ngd_curve = ngd_curve
        

    return (best_ngd_config, best_ngd_curve, best_diff)

#### grid search only ################
def grid_mapping(params, gdFile):
    seffMap = dict()
    ngd_mapped = []
    ts = 0.8
    tc = 0.8
    state = "Unstable"
    ### !!! CHANGE THE FILE NAME IF RUNNING FOR ALL !!!!!!
    # f_out = open(f"h{params['h']}_grid_G{gdFile}_stable.txt", 'w')
    f_out = open(f"q_mapped_s/h{params['h']}_s{ts}_c{tc}_grid_G{gdFile}_unstable.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h) in NGD population (S_Effective)\n")

    ### LOADING NECESSARY RESULTS FOR MAPPING 
    stabilityRes = loadStability(params['h'])
    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    ngd_results1 = load_pickle(f"new_allngdres001G_h5.pickle")
    ngd_results2 = load_pickle(f"new_allngdres001G_h.pickle")
    ngd_results = ngd_results1 | ngd_results2

    # print("STABILITY UNSTABLE REGIME:", stabilityRes[state])


    for (s, c, h) in gd_configs:

        # compute eq for stable regime:
        params_eq = {'config': (s, c, h), 'conversion': 'gametic'}
        eq = get_eq(params_eq)['q3']

        # if ((s, c, h), eq) in stabilityRes[state]:
        if ((s, c, h), eq) in stabilityRes[state] and math.isclose(s, ts) and math.isclose(c, tc):
        # if gd_res[(s,c,h)]['state'] == 'fix':
            print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(s, c, h)
            gd_curve = gd_res[(s, c, h)]['q']
            all_best_diff = dict()
            best_ngd_config = dict()
            best_ngd_curve = dict()

            output_folder = f"gd_candidates"
            outtxtName = os.path.join(output_folder, f"gd{s}_{c}_{h}_unstable_MSE.txt")
            if state == "Stable" or state == "Unstable":
                # for each gd config, open a new file to store the MSE for each s_ngd
                out1 = open(outtxtName, "w")
                if math.isclose(eq, 0.5):
                    continue
                h_ngd = eq/(2*eq-1)
                # print('----h_ngd----', h_ngd)
                # print("STABLE MAPPING WITH H_ngd =", h_ngd)
                if state == 'Unstable':
                    q_init_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]
                else:
                    q_init_list = [0.001]
                for q_ngd in q_init_list:
                    out1.write(f"q_init = {q_ngd:.4f}---------------------\n")
                    best_diff = 10000
                    s_mse_map = dict()
                    s_range = np.arange(-10.0, 10.0, 0.01)

                    ### plot the lambda vs. s_ngd curve
                    # plot_lambda_curve(h_ngd, eq, (s, c, h))
                
                    for s_ngd in s_range:
                        
                        curr_ngd_config, curr_ngd_curve, curr_MSE = find_candidate(s_ngd, h_ngd, q_ngd, state, (s, c, h), gd_curve, eq, s_mse_map)
                        # if math.isclose(s, 0.3) and math.isclose(c, 0.2):
                        #     print("SINGLE S RESULT", s_ngd, curr_ngd_config, curr_MSE)
                        # print(curr_ngd_config, curr_MSE)
                        if best_ngd_config == dict() or (curr_ngd_config != None and curr_MSE < best_diff):
                            best_diff = curr_MSE
                            best_ngd_config[q_ngd] = curr_ngd_config
                            best_ngd_curve[q_ngd] = curr_ngd_curve
                        # print(q_ngd, best_diff)
                        all_best_diff[q_ngd] = best_diff
                    
                        ### get delta_lambda
                        # delta_lambda = get_delta_lambda((s, c, h), best_ngd_config[q_ngd], eq)
                        # for each s_ngd, store the MSE and delta lambda
                    out1.write(f"q_init={q_ngd:.4f}: s_ngd={s_ngd:.4f}, MSE={best_diff:.6f}\n")
            # else: 
            #     for ngd_key, ngd_curve in ngd_results.items():
            #         diff = euclidean(ngd_curve, gd_curve)
            #         # print(diff)
            #         if diff < best_diff:
            #             best_diff = diff
            #             best_ngd_config = ngd_key
            #             best_ngd_curve = ngd_curve

            if best_ngd_config == None: 
                print("!!!!!!!!!!!!!NO STABLE SNGD")
                continue


            seffMap[(s, c, h)] = best_ngd_config
            ngd_mapped.append(best_ngd_curve)

            print(seffMap[(s, c, h)])
            
            for qkey in best_ngd_config.keys():
                # f_out.write(f"q_init = {qkey:.4f}")
                best_config = best_ngd_config[qkey]
                best_mse = all_best_diff[qkey]

                f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\tq_init={qkey:.4f}, s={'%.3f' % float(best_config[0])}, h={'%.3f' % float(best_config[1])}\terror={'%.6f' % float(best_mse)}\n")

            output_folder = "gd_candidates"
            # pickle_filename = os.path.join(output_folder, f"gd{(s, c, h)}_stable_finds.pickle")
            # Save the results dictionary into a pickle file.
            # save_pickle(pickle_filename, s_mse_map)
            # print(s_mse_map)
    
    # print(seffMap)
    
    return {'map': seffMap, 'ngC': ngd_mapped}
        

#### grid search for fixation regimes only ################
def grid_mapping_fix(params, gdFile):
    seffMap = dict()
    ngd_mapped = []
    f_out = open(f"h{params['h']}_grid_G{gdFile}_fix.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h) in NGD population (S_Effective)\n")

    stabilityRes = loadStability(params['h'])
    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    ngd_results = load_pickle(f"allngdres001G.pickle")

    for (s, c, h) in gd_configs:
        if ((s, c, h), 1.0) in stabilityRes['Fixation']:
        # if gd_res[(s,c,h)]['state'] == 'fix':
            print(s, c, h)
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
def gradient_mapping(params, gdFile):
    seffMap = dict()
    wms = []
    f_out = open(f"h{params['h']}_grad_G{gdFile}_fix.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h)in NGD population\n")

    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = loadStability(params['h'])
    grid_results = load_pickle(f"h{params['h']}_grid{gdFile}_G_fix.pickle")

    for (s, c, h) in gd_configs:
        if math.isclose(s, 0.2) and math.isclose(c, 0.9):
        # if ((s, c, h), 1.0) in stabilityRes['Fixation']:
        # if gd_res[(s,c,h)]['state'] == 'fix':
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s, best_h = grid_results['map'][(s,c,h)][0], grid_results['map'][(s,c,h)][1]
            # print('before gradient', best_s)
            best_s = -0.1
            best_h = h

            best_vars = gradient(best_s, best_h, s, c, h, params, gdFile)
            best_s, best_h = best_vars[0], best_vars[1]
            # best_s = optimization_s(best_s, s, c, h, params)
            # best_h = optimization(best_s, s, c, h, params)

            best_curve = wm(best_s, best_h, params['target_steps'], params['q0'])['q']
            best_diff = euclidean(best_curve, gd_curve)
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}, h={'%.3f' % best_h}\terror={'%.3f' % best_diff}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = (round(float(best_s), 3), round(float(best_h), 3))
            wms.append(best_curve)
    return {'map': seffMap, 'ngC': wms}