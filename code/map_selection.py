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

#### run_model: takes in params (with a fixed sch), runs a single gene drive simulation, returns a dictionary of q arrays and wbar array
def run_model(params):
    s = params['s']
    c = params['c']
    h = params['h']
    ts = params['target_steps']
    state = None
    # s_c = (1 - h * s) * c ## Gametic on modelrxiv
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
        freqs[t+1] = (curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (s_c + s_n)) / w_bar
        # wtfreqs[t+1] = (curr_p**2 + 2 * curr_p * (1 - curr_p) * s_n) / w_bar
        wtfreqs[t+1] = 1-freqs[t+1]

        if math.isclose(freqs[t+1], 1) or math.isclose(freqs[t+1], 0) or math.isclose(curr_q, freqs[t+1], rel_tol=1e-5):
            final = t+1
            break
        # if not math.isclose(freqs[t+1] + wtfreqs[t+1], 1.0):
        #     print(freqs[t+1], wtfreqs[t+1], t)
    # print(freqs, wtfreqs)
    if freqs[final-1] >= 0.99: state = 'fix'
    elif freqs[final-1] <= 0.01: 
        if freqs[final-1] > freqs[final-2]:
            state = 'fix'
        else: state = 'loss'
    elif at_eq(freqs): state = 'stable'
    else: 
        print('un', s, c, h)
        state = 'unstable'

    if state == None: print('none', freqs[t], freqs[t+1])

    return {'q': freqs[:final], 'p': wtfreqs[:final], 'w_bar': w[:final-1], 'state': state}

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
def plot_ngd(s, steps, init):
    with open('allngdres.pickle', 'rb') as f1:
        ngd_res = pickle.load(f1)
    # params = {'s': 0.6, 'c': 1, 'h': 0.5, 'target_steps': 100, 'q0': 0.1}
    # res = run_model(params)
    wt, mut = res['p'], res['q']
    plt.plot(wt, color = 'orange', label = 'wild-type')
    plt.plot(mut, color = 'blue', label = 'mutant')
    plt.ylabel('Allele Frequency')
    plt.xlabel('Time')
    plt.title('Allele frequency dynamics in non-gene drive population')
    plt.grid(True)
    plt.legend(title='Allele', bbox_to_anchor=(0.8, 0.5), loc='center left')
    plt.show()


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
    minVal, maxVal, step = 0.1, 1.0, 0.2
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
    for (s, c, h) in configs:
        # print('gd:', (s, c, h))
        params['s'], params['c'], params['h'] = s, c, h
        gd_res = run_model(params)
        gd_results[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = gd_res
    
    return configs, gd_results
        

def plot_gd(ts, tc):
    colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']
    with open('allgdres001.pickle', 'rb') as f1:
        gd_result = pickle.load(f1)
    configs, res = gd_result[0], gd_result[1]
    # print(configs)
    # print(res)

    plt.figure(figsize = (15, 7))
    for i in range(len(configs)):
        s, c, h = configs[i]
        # gd_color = cmap(1.*i/len(configs))
        cmap = plt.get_cmap(colormaps[int(s*10-1)])
        gd_color = cmap(c)
        time = np.arange(len(res[(s, c, h)]['q']))
        # if len(res[(s, c, h)]['q']) < 1500 and not math.isclose(res[(s, c, h)]['q'][-1], 1.0) and res[(s, c, h)]['q'][-1] < 0.1:
        if math.isclose(s, ts) and math.isclose(c, tc):
            print((s, c, h))
            print(res[(s, c, h)]['q'])

            plt.plot(time, res[(s, c, h)]['q'], color = gd_color, label = f"s = {s}, c = {c}, h = {h}")
    plt.ylabel('Gene Drive Allele Frequency')
    plt.xlabel('Time')
    plt.title("Change in Mutant Allele Frequency")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.grid(True)
    plt.legend(title='population condition', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.show()

# plot_gd(s, c)

### pickle dump ngd results: {(s,h): curve} and gd results [configlist, gd_res]
def getcurves(params):
    wm_results = dict()
    print(s_range, h_range)
    for s_nat in s_range:  
        for h_nat in h_range: 
            wm_curve = wm(s_nat, h_nat, params['target_steps'], params['q0'])['q']
            wm_results[(round(s_nat, 3), round(h_nat, 3))] = wm_curve
    # gd_configs, gd_res = gd_simulation(params)
    # gd_results = [gd_configs, gd_res]  # gd_configs: list of tuples, gd_res: map {config: curve}
    # print('length of configs', len(gd_configs))
    with open('allngdres.pickle', 'wb') as f1:
        pickle.dump(wm_results, f1)
    # with open('allgdres001gametic.pickle', 'wb') as f2:  ### allgdres001: step = 0.01
    #     pickle.dump(gd_results, f2)

minVal, maxVal, step = -1, 1, 0.01
s_range = np.arange(minVal, maxVal, step)
h_range = np.arange(0, maxVal, step)
colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']

# print(wm_results[-1.0])

# get f(h) from sum of squared difference between points on two curves
def f(ngd_s, ngd_h, s, c, h, params):
    with open('allgdres.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    gd_curve = gd_res[(s, c, h)]['q']

    ngd_curve = wm(ngd_s, ngd_h, params['target_steps'], params['q0'])['q']
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

def opt_r(F, grad_F, vars, s, c, h, params, alpha=0.01, beta=0.5, initial_r=1):
    r = initial_r
    while True:
        new_vars = vars - r * grad_F
        new_F = f(new_vars[0], new_vars[1], s, c, h, params)
        if  new_F <= F - alpha * r * np.dot(grad_F, grad_F):
            return r
        r *= beta

def gradient(s, c, h, params):
    ngd_s = s
    ngd_h = h
    vars = np.array([ngd_s, ngd_h])
    delta = 0.001
    iterations = 1000
    grad_F = np.zeros(len(vars))
    F = 0

    for i in range(iterations):
        print('iteration', i, s, c, h)
        F = f(vars[0], vars[1], s, c, h, params)
        for var_i in range(len(vars)):
            vars_plus_delta = vars.copy()
            vars_plus_delta[var_i] += delta
            f_s1 = f(vars_plus_delta[0], vars_plus_delta[1], s, c, h, params)
            df_s = (f_s1 - F)/delta
            grad_F[var_i] = df_s

        
        # minimalize f((s, h) - r*grad(s, h))
        r = opt_r(F, grad_F, vars, s, c, h, params)
        new_vars = vars - r*grad_F
        print('1st', F, new_vars)

        # if new_vars[1] < 0 or new_vars[1] > 1:
        #     print(vars)
        #     break

        new_vars[0] = np.clip(new_vars[0], -100, 1) 
        new_vars[1] = np.clip(new_vars[1], 0, 1)  # Constraint on 'h'
        if math.isclose(f(new_vars[0], new_vars[1], s, c, h, params), F, rel_tol=1e-5):
            break
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
def mapping(params):
    seffMap = dict()
    wms = []
    f_out = open('outputs/s_map.txt', 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h)in well-mixed population\n")

    with open('allgdres.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    print('params', gd_configs)

    with open('allngdres.pickle', 'rb') as f1:
        wm_results = pickle.load(f1)

    for (s, c, h) in gd_configs:
        if math.isclose(s, 0.5):
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s = None
            best_h = h
        
            for ngd_key, ngd_curve in wm_results.items():
                diff = euclidean(ngd_curve, gd_curve)
                if diff < best_diff:
                    best_diff = diff
                    best_config = ngd_key
                    
        
        # gradient descent or newton_raphson

            best_vars = gradient(s, c, h, params)
            best_s, best_h = best_vars[0], best_vars[1]
            # best_s = optimization_s(best_s, s, c, h, params)
            # best_h = optimization(best_s, s, c, h, params)

            best_curve = wm(best_s, best_h, params['target_steps'], params['q0'])['q']
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}, h={'%.3f' % best_h}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = (round(best_s, 3), round(float(best_h), 3))
            wms.append(best_curve)
    return {'map': seffMap, 'ngC': wms}

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
    gd_s = 0.6
    gd_c = 0.4
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
    with open('allgdres001.pickle', 'rb') as f2:
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

    print(states)
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

# mapping curves of ngd and gd populations ###########
def plot_mapping(params):
    with open('allgdres.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    with open('gradient_fix_mapping_results.pickle', 'rb') as f:
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
            gd_color = cmap(s)
            best_s, best_h = sMap[(s, c, h)][0], sMap[(s, c, h)][1]
            cmap1 = plt.get_cmap('PuRd')
            wm_curve = wms[i]
            # print(wm_curve)
            w_color = cmap1(abs(best_s))
            time1 = np.arange(0, len(wm_curve))
            time2 = np.arange(0, len(gd_res[(s, c, h)]['q']))
            time3 = np.arange(0, len(gd_res[(s, c, h)]['q'])-1)
            plt.plot(time1, wm_curve, marker = 'o', color = w_color, markersize=3, linestyle = '-', label = f'non_gd s = {best_s}, non_gd h = {best_h}')
            plt.plot(time2, gd_res[(s, c, h)]['q'], color = gd_color, label = f"s = {s}, c = {c}, h = {h}")
            # plt.plot(time3, gd_res[(s, c, h)]['w_bar'], color = 'r', label = f"wbar for s = {s}, c = {c}, h = {h}")
            # plt.plot(time, gd_curve, marker = 'o', color = g_color, linestyle = '-', label = f'gene drive model s = {round(s, 2)}')

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
    # mapping_result = grid_mapping(params) # CHANGE THIS TO SWITCH MAPPING METHOD
    # print('Running and saving the mapping results...')
    # with open('grid_loss_mapping_results.pickle', 'wb') as f:
    #     pickle.dump(mapping_result, f)

    plot_mapping(params)
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

