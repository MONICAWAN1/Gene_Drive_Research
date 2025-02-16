import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import math
import pickle

def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# none gene drive model
def wm(s, h, target_steps, q_init):
    q_freqs = np.zeros(target_steps)
    p_freqs = np.zeros(target_steps)
    q_freqs[0] = q_init # mutant
    p_freqs[0] = 1 - q_init
    w = np.zeros(target_steps)
    final = target_steps

    for t in range(target_steps-1):
        # freqs[t+1] = freqs[t] + s * freqs[t] * (1-freqs[t])
        curr_q = q_freqs[t]
        curr_p = p_freqs[t]
        w_bar = curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (1-h*s) + (1 - curr_q)**2
        w[t] = w_bar
        q_freqs[t+1] = (curr_q**2 * (1 - s) + curr_q * (1 - curr_q) * (1 - h * s)) / w_bar
        p_freqs[t+1] = (curr_p**2 + curr_p * (1 - curr_p) * (1 - h * s)) / w_bar

        if math.isclose(q_freqs[t+1], 1) or math.isclose(q_freqs[t+1], 0) or math.isclose(curr_q, q_freqs[t+1], rel_tol=1e-5):
            final = t+1
            break
    return {'p': p_freqs[:final], 'q': q_freqs[:final], 'w_bar': w[:final-1]}


#ngd model
def haploid_se(params):
    s,c,h = params['s'], params['c'], params['h']
    # s,c,h = 0.4, 0.6, 0
    # s = 1-c+c*s*(2-h)-h*s # zygotic
    s = h*s-c+c*h*s  #gametic se
    ts = params['target_steps']
    freqs = np.zeros(params['target_steps'])
    wtfreqs = np.zeros(params['target_steps'])
    w = np.zeros(params['target_steps'])
    freqs[0] = params['q0']
    wtfreqs[0] = 1- params['q0']
    final = ts
    for t in range(ts-1):
        curr_q = freqs[t] # mutant
        curr_p = wtfreqs[t] # wildtype
        w_bar = curr_q*(1-s) + curr_p
        w[t] = w_bar

        freqs[t+1] = curr_q*(1-s)/w_bar
        wtfreqs[t+1] = 1-freqs[t+1]

        if freqs[t+2] > 1 or math.isclose(freqs[t+1], 1) or math.isclose(freqs[t+1], 0) or math.isclose(curr_q, freqs[t+1], rel_tol=1e-5):
            final = t+2
            break

    state = checkState(final, freqs, params)
    if state == None: print('none', freqs[t], freqs[t+1])

    return {'q': freqs[:final], 'p': wtfreqs[:final], 'w_bar': w[:final-1], 'state': state}

# ngd haploid without se
def haploid(params):
    s = params['s']
    # s = 1-c+c*s*(2-h)-h*s
    ts = params['target_steps']
    freqs = np.zeros(params['target_steps'])
    wtfreqs = np.zeros(params['target_steps'])
    w = np.zeros(params['target_steps'])
    freqs[0] = params['q0']
    wtfreqs[0] = 1- params['q0']
    final = ts
    for t in range(ts-1):
        curr_q = freqs[t] # mutant
        curr_p = wtfreqs[t] # wildtype
        w_bar = curr_q*(1-s) + curr_p
        w[t] = w_bar

        freqs[t+1] = curr_q*(1-s)/w_bar
        wtfreqs[t+1] = 1-freqs[t+1]

        if math.isclose(freqs[t+1], 1) or math.isclose(freqs[t+1], 0) or math.isclose(curr_q, freqs[t+1], rel_tol=1e-5):
            final = t+1
            break

    state = checkState(final, freqs, params)
    if state == None: print('none', freqs[t], freqs[t+1])

    return {'q': freqs[:final], 'p': wtfreqs[:final], 'w_bar': w[:final-1], 'state': state}


#### run_model: takes in params (with a fixed sch), runs a single gene drive simulation, returns a dictionary of q arrays and wbar array
def run_model(params):
    s = params['s']
    c = params['c']
    h = params['h']
    ts = params['target_steps']
    state = None
    # s_c = (1 - s) * c ## zygotic in paper
    s_c = (1 - h*s) * c ## gametic
    s_n = 0.5 * (1 - h * s) * (1 - c) # when c = 1, h doesn't matter 
    wtfreqs = np.zeros(params['target_steps'])
    freqs = np.zeros(params['target_steps'])
    w = np.zeros(params['target_steps'])
    freqs[0] = params['q0']
    wtfreqs[0] = 1 - params['q0']
    final = ts

    for t in range(ts-1):
        curr_q = freqs[t] # mutant
        curr_p = wtfreqs[t] # wildtype
        w_bar = curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (s_c + 2 * s_n) + (1 - curr_q)**2
        w[t] = w_bar

        ## original gene drive model
        freqs[t+1] = (curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (s_c + s_n)) / w_bar
        wtfreqs[t+1] = (curr_p**2 + 2 * curr_p * (1 - curr_p) * s_n) / w_bar

        ## approximation
        # se = 1-c+c*s*(2-h)-h*s
        # freqs[t+1] = 2 * curr_q * se
        # wtfreqs[t+1] = 1-freqs[t+1]


        if math.isclose(freqs[t+1], 1) or math.isclose(freqs[t+1], 0) or math.isclose(curr_q, freqs[t+1], rel_tol=1e-5):
            final = t+1
            break
        # if not math.isclose(freqs[t+1] + wtfreqs[t+1], 1.0):
        #     print(freqs[t+1], wtfreqs[t+1], t)
    # print(freqs, wtfreqs)
    state = checkState(final, freqs, params)
    if state == None: print('none', freqs[t], freqs[t+1])

    return {'q': freqs[:final], 'p': wtfreqs[:final], 'w_bar': w[:final-1], 'state': state}

def checkState(final, freqs, params):
    if freqs[final-1] >= 0.99: state = 'fix'
    elif freqs[final-1] <= 0.01: 
        if freqs[final-1] > freqs[final-2]:
            state = 'fix'
        else: state = 'loss'
    elif at_eq(freqs): state = 'stable'
    else: 
        state = 'unstable'
    return state

def at_eq(freqs):
    differences = [abs(freqs[i+1] - freqs[i]) for i in range(len(freqs)-1)]
    for diff in differences[-10:]:
        if diff >= 0.001:
            return False
    return True

#### Euclidean distance formula #############################
def euclidean(result1, result2):
    diff1 = diff2 = 0
    q0 = result1
    q0_m = result2
    if len(q0) < len(q0_m):
        q0 = np.append(q0, [q0[-1]] * (len(q0_m) - len(q0)))
    else:
        q0_m = np.append(q0_m, [q0_m[-1]] * (len(q0) - len(q0_m)))

    for i in range(len(q0)):
        diff1 += (q0[i]-q0_m[i])**2
    return diff1**0.5

### plot simple dynamics for gd and non gd ################
def plot_ngd(params):
    with open('pickle/allhaploidres.pickle', 'rb') as f1:
        ngd_res = pickle.load(f1)
    
    res = haploid(params)
    print(res)
    
    wt, mut, w_bar = res['p'], res['q'], res['w_bar']
    plt.plot(wt, color = 'orange', label = 'wild-type')
    plt.plot(mut, color = 'blue', label = 'mutant')
    # plt.plot(w_bar, color = 'r', label = 'w_bar')
    plt.ylabel('Allele Frequency')
    plt.xlabel('Time')
    plt.title('Allele frequency dynamics in haploid non-gene drive population')
    plt.grid(True)
    plt.legend(title='Allele', bbox_to_anchor=(0.8, 0.5), loc='center left')
    plt.show()

# plot_ngd()


def delta_curve(curve):
    first_d = []
    for t in range(len(curve)-1):
        delta = curve[t+1] - curve[t]
        first_d.append(delta)
    return first_d

def derivative_plot(params, gd_configs, fitting_res):
    plt.figure(figsize=(10,7))
    for (s, c, h) in gd_configs:
        params['s'] = s
        params['c'] = c
        params['h'] = h
        res = run_model(params)
        gd_allele, wt_allele = res['q'], res['p']
    
        d1 = delta_curve(gd_allele)

        second_d1 = delta_curve(d1)
        cmap1 = plt.get_cmap('BuPu')
        gd_color = cmap1(c)
        time = list(range(params['target_steps']-2))
        plt.plot(time, second_d1, color = gd_color, label = f'gene drive s={round(s, 3)}, c={round(c, 3)}, h={round(h, 3)}')

        fitting_s = fitting_res[(s, c, h)][0]
        for h in np.arange(0, 1, 0.1):
            ngd_curve = wm(fitting_s, h, params['target_steps'], params['q0'])['q']
            ngd_d1 = delta_curve(ngd_curve)
            ngd_d2 = delta_curve(ngd_d1)

            cmap2 = plt.get_cmap('Oranges')
            ngd_color = cmap2(h)
            time2 = time[:len(ngd_d2)]
        # plt.plot(time, ngd_d2, color = ngd_color, label = f'c={round(c, 3)}, s_eff={fitting_s}')
            plt.plot(time2, ngd_d2, color = ngd_color, label = f'no gene drive s={round(float(fitting_s), 3)}, h={round(h, 3)}')
    
    # plt.plot(time, wt_allele, color = 'orange')

    plt.legend(title='Parameters', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.xlabel('time', fontsize = 15)
    plt.ylabel('second derivative of gene-drive allele curve', fontsize = 15)
    plt.show()

# test_params = {'n': 200, 'target_steps': 200, 'q0': 0.1, 's': 0.45, 'c': 0.6, 'h': 0.5}
# derivative_plot(test_params)



### run simulation for a list of configurations, returns a list of configs and a dictionary {config: curve} #############
def gd_simulation(params):
    minVal, maxVal, step = 0.0, 1.0, 0.1
    configs = []
    s_vals = np.arange(minVal, maxVal, 0.01)
    c_vals = np.arange(0, maxVal, 0.01)
    h_vals = np.arange(0, maxVal, 0.01)

    # get a list of configurations for simulations 
    for s in s_vals:
        if s>0:
            for c in c_vals:
                if c>0:
                    for h in h_vals:
                        if math.isclose(h, 0):
                            configs.append((round(float(s), 3), round(float(c), 3), round(float(h), 3)))

    gd_results = dict()
    hs_results = dict()
    for (s, c, h) in configs:
        # print('gd:', (s, c, h))
        params['s'], params['c'], params['h'] = s, c, h
        gd_res = run_model(params)
        hs_res = haploid_se(params)
        gd_results[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = gd_res
        hs_results[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = hs_res
    
    return configs, gd_results, hs_results
        

def plot_gd(ts, tc):
    colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']
    with open('pickle/allgdres001.pickle', 'rb') as f1:
        gd_result1 = pickle.load(f1)
    configs, res1 = gd_result1[0], gd_result1[1]
    # print(configs)
    # print(res)

    with open('pickle/approxres001.pickle', 'rb') as f2:
        gd_result2 = pickle.load(f2)
    configs2, res2 = gd_result2[0], gd_result2[1]    

    plt.figure(figsize = (15, 7))
    for i in range(len(configs2)):
        s, c, h = configs2[i]
        # gd_color = cmap(1.*i/len(configs))
        # cmap = plt.get_cmap(colormaps[int(s*10-1)])
        cmapGD = plt.get_cmap(colormaps[2])
        cmapAP = plt.get_cmap(colormaps[4])
        gd_color = cmapGD(c)
        ap_color = cmapAP(c)

        # if len(res[(s, c, h)]['q']) < 1500 and not math.isclose(res[(s, c, h)]['q'][-1], 1.0) and res[(s, c, h)]['q'][-1] < 0.1:
        if math.isclose(s, ts) and math.isclose(c, tc):
            print((s, c, h))
            print(res1[(s, c, h)]['q'])
            time1 = np.arange(len(res1[(s, c, h)]['q']))
            time2 = np.arange(len(res2[(s, c, h)]['q']))

            plt.plot(time1, res1[(s, c, h)]['q'], color = gd_color, label = f"s = {s}, c = {c}, h = {h}, original")
            plt.plot(time2, res2[(s, c, h)]['q'], color = ap_color, label = f"s = {s}, c = {c}, h = {h}, approximation")
    plt.ylabel('Gene Drive Allele Frequency')
    plt.xlabel('Time')
    plt.title("Comparison graph")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.grid(True)
    plt.legend(title='population condition', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.show()

# plot_gd(0.4, 0.6)

### pickle dump ngd results: {(s,h): curve} and gd results [configlist, gd_res]
def getcurves(params):
    hapRes = dict()
    print('s_range', s_range, h_range)

    for s_hap in s_range:
        params['s'] = s_hap
        haploidC= haploid(params)['q']
        hapRes[float(round(s_hap, 3))] = haploidC

    gd_configs, gd_res, hs_res = gd_simulation(params)
    gd_results = [gd_configs, gd_res]  # gd_configs: list of tuples, gd_res: map {config: curve}
    hs_results = [gd_configs, hs_res]
    print('length of configs', len(gd_configs))
    with open('allgdres001G.pickle', 'wb') as f1:
        pickle.dump(gd_results, f1)

    with open('allhapseres001G.pickle', 'wb') as f2:
        pickle.dump(hs_results, f2)

    # with open('allhaploidres.pickle', 'wb') as f3:
    #     pickle.dump(hapRes, f3)
    # # with open('approxres001.pickle', 'wb') as f2:  ### allgdres001: step = 0.01
    # #     pickle.dump(gd_results, f2)

minVal, maxVal, step = -30, 1, 0.01
s_range = np.arange(minVal, maxVal, step)
h_range = np.arange(0, maxVal, step)
colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']

# print(wm_results[-1.0])

# get f(h) from sum of squared difference between points on two curves
def f(ngd_s, s, c, h, params):
    # ngd_h = float(round(ngd_h, 3))
    with open('pickle/allgdres.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    gd_curve = gd_res[(s, c, h)]['q']

    ## get ngd curve to find the current distance
    # ngd_curve = wm(ngd_s, ngd_h, params['target_steps'], params['q0'])['q']
    params['s'] = ngd_s
    # print('ngds', ngd_s)
    ngd_curve = haploid(params)['q']
    f = 0

    if len(gd_curve) < len(ngd_curve):
        gd_curve = np.append(gd_curve, [gd_curve[-1]] * (len(ngd_curve) - len(gd_curve)))
    else:
        ngd_curve = np.append(ngd_curve, [ngd_curve[-1]] * (len(gd_curve) - len(ngd_curve)))
    for i in range (len(gd_curve)):
        f += (gd_curve[i] - ngd_curve[i]) ** 2
    return f**0.5

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

def opt_r(F, grad_F, vars, s, c, h, params, alpha=1e-4, beta=0.5, r_init=1.0):
    r = r_init
    while True:
        new_vars = vars - r * grad_F
        new_F = f(new_vars[0], new_vars[1], s, c, h, params)
        if  new_F <= F - alpha * r * np.dot(grad_F, grad_F):
            return r
        r *= beta

def gradient(startS, s, c, h, params):
    ngd_s = startS
    vars = np.array([ngd_s]) 
    delta = 0.0001
    iterations = 1000
    grad_F = np.zeros(1)
    F = 0
    prev_F = F+10

    for i in range(iterations):
        print('iteration', i, s, c, h)
        F = f(vars[0], s, c, h, params)
        if abs(F - prev_F) < 1e-8:
            print("Stopping: function change is too small.")
            break
    
        vars_plus_delta = vars.copy()
        vars_plus_delta[0] += delta
        print(vars_plus_delta)
        f_s1 = f(vars_plus_delta[0], s, c, h, params)
        df_s = (f_s1 - F)/delta
        print('df_s', df_s)
        grad_F[0] = df_s

        
        # minimalize f((s, h) - r*grad(s, h))
        r = opt_r(F, grad_F, vars, s, c, h, params)
        new_vars = vars - r*grad_F
        print('1st', F, new_vars)

        # if new_vars[1] < 0 or new_vars[1] > 1:
        #     print(vars)
        #     break
        prev_F = F

        new_vars[0] = np.clip(new_vars[0], -100, 1) 
        # if math.isclose(f(new_vars[0], s, c, h, params), F, rel_tol=1e-5):
        #     print("too close!", f(new_vars[0], s, c, h, params))
        #     print('break')
        #     break
        vars = new_vars
        print('2nd', F, vars)
    return vars


#### grid search only ################
def grid_mapping(params):
    seffMap = dict()
    ngd_mapped = []
    f_out = open('outputs/grid_mapping2.txt', 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h) in well-mixed population (S_Effective)\n")

    with open('allgdres.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    with open('allngdres.pickle', 'rb') as f1:
        ngd_results = pickle.load(f1)

    for (s, c, h) in gd_configs:
        if gd_res[(s, c, h)]['state'] == 'loss':
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
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_ngd_config[0]}, h={'%.3f' % best_ngd_config[1]}\n")
    
    # print(seffMap)
    
    return {'map': seffMap, 'ngC': ngd_mapped}
        


#### grid + gradient ########################
def hap_mapping(params):
    seffMap = dict()
    hapMappedCurves = []
    f_out = open('outputs/hap_G_fix_map.txt', 'w')
    f_out.write(f"Gene Drive Configuration\t\ts in haploid ngd population\n")

    with open('pickle/allgdresG.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    # print('params', gd_configs)

    with open('pickle/allhaploidres.pickle', 'rb') as f1:
        hap_results = pickle.load(f1)

    for (s, c, h) in gd_configs:
        # only testing for s = 0.4, c=0.6
        # if math.isclose(s, 0.4) and math.isclose(c, 0.6):
        if s < c:
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s = None
            best_h = h
        
            for ngd_key, ngd_curve in hap_results.items():
                diff = euclidean(ngd_curve, gd_curve)
                if diff < best_diff:
                    best_diff = diff
                    best_s = ngd_key
                    
        
        # gradient descent or newton_raphson
            print('before gradient:', best_s)

            # best_vars = gradient(best_s, s, c, h, params)
            # best_s = best_vars[0]
            # best_s = optimization_s(best_s, s, c, h, params)

            ### haploid
            hap_params = {'s': best_s, 'q0': params['q0'], 'target_steps': params['target_steps']}
            best_curve = haploid(hap_params)['q']
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = round(best_s, 3)
            hapMappedCurves.append(best_curve)
    return {'map': seffMap, 'ngC': hapMappedCurves}

def getdiff():
    with open('allgdres.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    with open('allngdres.pickle', 'rb') as f1:
        wm_results = pickle.load(f1)

    diffmap = dict()
    ngd_config = []
    s = 0.6
    c = 0.4
    h = 0

    gd_curve = gd_res[(s, c, h)]['q']
    print('found', gd_curve)
    for ngd_key, ngd_curve in wm_results.items():
        diff = euclidean(ngd_curve, gd_curve)
        diffmap[ngd_key] = diff
        ngd_config.append(ngd_key)
    
    with open('mappingdiff.pickle', 'wb') as f:
        pickle.dump(diffmap, f)

def ploterror():
    gd_s = 0.4
    gd_c = 0.6
    gd_h = 0.0
    with open('mappingdiff.pickle', 'rb') as f:
        diffmap = pickle.load(f)
    
    # print(diffmap)
    ngd_s = sorted(set([conf[0] for conf in diffmap.keys()]))
    ngd_h = sorted(set([conf[1] for conf in diffmap.keys()]))

    # print(ngd_h, len(ngd_s), len(ngd_h))
    errors = np.zeros((len(ngd_h), len(ngd_s)))

    for (s, h), error in diffmap.items():
        i = ngd_h.index(h)  # Row index (h axis)
        j = ngd_s.index(s)  # Column index (s axis)
        errors[i, j] = error
    
    print(errors)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(errors, aspect='auto', cmap='BuPu', origin='lower',
               extent=[min(ngd_s), max(ngd_s), min(ngd_h), max(ngd_h)])
    plt.colorbar(label='Euclidean Distance (Error of mapping)')
    plt.xlabel('s in non-gene-drive')
    plt.ylabel('h in non-gene-drive')
    plt.title(f"Error Heatmap for the Mapping from Gene-Drive configuration {(gd_s, gd_c, gd_h)} to Non-Gene-Drive Model")
    plt.show()
    

def plot_gdtn(params):   # the dense mesh figure
    # mapping_result = mapping(params)
    with open('sch_to_s_results.pickle', 'rb') as f:
        map_results = pickle.load(f)
    
    with open('gd_simresults.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    seffmap, ngdcurves = map_results['map'], map_results['ngC']
    print(seffmap)
    gd_configs = seffmap.keys()
    slist = sorted(set([conf[0] for conf in gd_configs]))
    clist = sorted(set([conf[1] for conf in gd_configs]))
    hlist = sorted(set([conf[2] for conf in gd_configs]))
    seff = [seffmap[config] for config in gd_configs]
    plt.figure(figsize = (15, 7))
    cmap = plt.get_cmap('BuPu')
    for s in slist:
        sefflist = [seffmap[(s,c,0.5)][0] for c in clist]
        # print(sefflist)
        cvcolor = cmap(s)
        plt.plot(clist, sefflist, marker = 'o', linestyle='-', color = cvcolor, label = f's = {s}')
    plt.ylabel('S_effective in non-gene-drive population')
    plt.xlabel('C in gene drive population')
    plt.legend(title='S in gene drive population', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.title("Map from gene drive configuration to effective S in non-gene-drive population (h = 0.5)")
    plt.show()


# sweep through all configurations, get final state
# partition graph (fixed h, x is c, y is s, color indicates s-eff/final state in gd simulation)
def partition(params):
    # loading
    # with open('sch_to_s_results.pickle', 'rb') as f:
    #     map_results = pickle.load(f)
    with open('approxres001.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    # seffmap, ngdcurves = map_results['map'], map_results['ngC']
    slist = sorted(set([conf[0] for conf in gd_configs]))
    clist = sorted(set([conf[1] for conf in gd_configs]))
    s_effs = []
    configurations = []
    states = []
    finals = []
    statemap = {'fix': 2.5, 'loss': 0.5, 'stable': 1.5}
    h = 0 # need to change this!
    for s in slist:
        for c in clist:
            configurations.append((s, c))
            # s_effs.append(seffmap[(s, c, h)][0])
            if gd_res[(s,c,h)]['state'] == 'unstable': plot_gd(s, c)
            states.append(statemap[gd_res[(s,c,h)]['state']])
            finals.append(gd_res[(s,c,h)]['q'][-1])

    s_values = [conf[0] for conf in configurations]
    c_values = [conf[1] for conf in configurations]

    # print(len(s_effs))
    # print(len(configurations))
    
    # Normalize S-effective values to map to colors
    norm = mcolors.Normalize(min(finals), max(finals))
    bounds = [0, 1, 2, 3]
    # cmap = plt.cm.get_cmap('Purples', len(bounds) - 1)
    cmap = plt.cm.get_cmap('viridis')
    # norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    scatter = ax.scatter(c_values, s_values, c=finals, cmap=cmap, s=50, norm=norm) 

    # cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 1, 2, 3])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Final Gene Drive Allele Frequency')
    # cbar.set_ticklabels(['', 'Loss', 'Equilibrium', 'Fixation'])

    ax.set_ylabel('Selection Coefficient (s)')
    ax.set_xlabel('Conversion Factor (c)')
    ax.set_title(f'Partition of Gene Drive Configurations by final gene drive allele frequency with H = {h}')

    # Show plot
    plt.show()


def plotMapDiff(params):
    with open('pickle/allgdres001G.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    # with open('pickle/hap_grid_gametic_fix_results.pickle', 'rb') as f:
    #     mapResult = pickle.load(f)

    with open('pickle/allhapseres001G.pickle', 'rb') as f: # the hap_se results
        mapResult = pickle.load(f)

    # print(mapResult)
    # sMap, wms = mapResult['map'], mapResult['q']
    gds = [res['q'] for res in gd_res.values()]

    valid_configs = []
    for (s, c, h) in gd_configs:
        if s < c:
            valid_configs.append((s, c, h))

    # Build sorted lists of s and c to index into a 2D array
    all_s = sorted(set(s for (s,c,h) in valid_configs))
    all_c = sorted(set(c for (s,c,h) in valid_configs))

    # Initialize a 2D array of differences
    diff_map = np.zeros((len(all_s), len(all_c)))

    config_index = {conf: i for i, conf in enumerate(gd_res.keys())}

    # Compute the difference for each (s, c, h) and place in diff_map
    for (s, c, h) in valid_configs:
        # The index of this config in wms
        idx = config_index[(s, c, h)]
        wm_curve = gds[idx]

        # plot haploid using Se
        paramSe = {'s':s, 'c':c, 'n': 500, 'h': 0, 'target_steps': 40000, 'q0': 0.001}
        hapSe = haploid_se(paramSe)['q']

        curveDiff = euclidean(wm_curve, hapSe)
        row_idx = all_s.index(s)
        col_idx = all_c.index(c)
        diff_map[row_idx, col_idx] = curveDiff

    X, Y = np.meshgrid(all_c, all_s)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X, Y, diff_map, shading='auto')
    plt.colorbar(label='Difference (mapped vs. hapSe)')
    plt.xlabel('c')
    plt.ylabel('s')
    plt.title('Difference Heatmap: GD vs. Haploid Se')
    plt.show()



# mapping curves of ngd and gd populations ###########
def plot_mapping(params):
    with open('pickle/allgdresG.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    with open('pickle/hap_grid_gametic_fix_results.pickle', 'rb') as f:
        mapResult = pickle.load(f)

    sMap, wms = mapResult['map'], mapResult['ngC']
    gd_configs = [key for key in sMap]
    print(gd_configs)

    plt.figure(figsize = (15, 7))

    for i in range(len(gd_configs)):
        s, c, h = gd_configs[i]
        if math.isclose(s, 0.4) and math.isclose(c, 0.6):
            # gd_color = cmap(1.*i/len(configs))
            cmap = plt.get_cmap(colormaps[i % len(colormaps)])
            gd_color = cmap(0.8)
            # best_s, best_h = sMap[(s, c, h)][0], sMap[(s, c, h)][1]
            best_s = float(sMap[(s, c, h)])
            cmap1 = plt.get_cmap('PuRd')
            wm_curve = wms[i]
            w_color = cmap1(abs(0.5))

            # plot haploid using Se
            paramSe = {'s':s, 'c':c, 'n': 500, 'h': 0, 'target_steps': 40000, 'q0': 0.001}
            hapSe = haploid_se(paramSe)['q']

            # nearby = []
            # for hnew in np.arange(0, 1, 0.1):
            #     newc = wm(best_s, hnew, params['target_steps'], params['q0'])['q']
            #     nearby.append(newc)
            #     time = np.arange(0, len(newc))
            #     plt.plot(time, newc, marker = 'o', color = w_color, markersize=3, linestyle = '-', label = f'haploid s = {best_s}, h = {hnew}')

            # print(wm_curve)
            time1 = np.arange(0, len(wm_curve))
            time2 = np.arange(0, len(gd_res[(s, c, h)]['q']))
            time_se = np.arange(0, len(hapSe))
            # time3 = np.arange(0, len(gd_res[(s, c, h)]['q'])-1)
            plt.plot(time1, wm_curve, marker = 'o', color = w_color, markersize=3, linestyle = '-', label = f'haploid s = {best_s}')
            # plt.plot(time_se, hapSe, marker = 'o', color = 'b', markersize=3, label = f"haploid using Se")
            plt.plot(time2, gd_res[(s, c, h)]['q'], color = gd_color, label = f"s = {s}, c = {c}, h = {h}")
            # plt.plot(time3, gd_res[(s, c, h)]['w_bar'], color = 'r', label = f"wbar for s = {s}, c = {c}, h = {h}")


    # param_text = f"orange: non-gene-drive population\nblue: population with gene drive\nq_initial = {params['q0']}"
    # plt.figtext(0.6, 0.2, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    plt.ylabel('Gene Drive/Mutant Allele Frequency', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.title("Dynamics in Mutant Allele Frequency overtime")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.grid(True)
    legend = plt.legend(title='population condition', bbox_to_anchor=(1, 1.05), loc='upper left')
    export_legend(legend)
    plt.show()

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def main():
    params = {'n': 500, 'h': 0, 'target_steps': 40000, 'q0': 0.001}

    ### if configurations are changed, then run the whole simulation and save the result

    # getcurves(params) # get all curves of gd and non-gd based on current set of configurations

    ### ploting the resulting state
    

    ### mapping gd to non-gd
    # mapping_result = hap_mapping(params) # CHANGE THIS TO SWITCH MAPPING METHOD
    # print(mapping_result)
    # print('Running and saving the mapping results...')
    # with open('hap_grid_gametic_fix_results.pickle', 'wb') as f:
    #     pickle.dump(mapping_result, f)
    
    plotMapDiff(params)
    # plot_mapping(params)
    # plot_gd(0.4, 0.6)
    # params['s'] = -0.4
    # plot_ngd(params)
    # partition(params)
    # getdiff()
    # ploterror()

    # with open('sch_to_s_results.pickle', 'rb') as f:
    #     map_results = pickle.load(f)
    
    # with open('gd_simresults.pickle', 'rb') as f2:
    #     gd_results = pickle.load(f2)
    # gd_configs, gd_res = gd_results[0], gd_results[1]


    # derivative_plot(params, gd_configs, map_results['map'])


if __name__ == '__main__':
    main()

