import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import math
import pickle

from models import run_model, wm, haploid_se, haploid
from utils import export_legend, euclidean, load_pickle, save_pickle

def loadGrad():
    return load_pickle(f"h0.0_hap_gradient_G_fix.pickle")

colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']

diffmap5 = load_pickle("h0.0_mappingdiff_hap_grid.pickle")
diffmap6 = load_pickle("h0.0_mappingdiff_gdhapse01.pickle")
# print(diffmap1, '\n', diffmap2)

# BuPu Color Scale
all_values = list(diffmap5.values()) + list(diffmap6.values()) + [0.011150530195789305]
min_val = min(all_values)
max_val = max(all_values)
print(min_val, max_val)

'''
Plot error vs h for a specific configuration
'''
def plot_errorh(h):
    hapse_errors = []
    hapgrid_errors = []
    s = 0.2
    c = 0.9
    h_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for h in h_range:
        hapseDiff = load_pickle(f"h{h}_mappingdiff_gdhapse01.pickle")
        hapgridDiff = load_pickle(f"h{h}_mappingdiff_hap_grid.pickle")
        # print(list(hapseDiff.keys()))
        hapse_errors.append(hapseDiff[(s, c, h)])
        hapgrid_errors.append(hapgridDiff[(s, c, h)])

    
    plt.figure(figsize=(8, 6))
    plt.plot(h_range, hapse_errors, label = "Haploid_Se Model")
    plt.plot(h_range, hapgrid_errors, label = "Grid Search Haploid")

    plt.xlabel('h', fontsize = 14)
    plt.ylabel('MSE between Gene-Drive curve and Mapped curve', fontsize = 14)
    plt.title(f'Change in Mapping Error over H at S = {s}, C = {c}')
    plt.legend(title='Mapping Method', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.tight_layout()

    plt.show()
        

### plot simple dynamics for gd and non gd ################
def plot_ngd(s, steps, init):
    with open('pickle/allngdres.pickle', 'rb') as f1:
        ngd_res = pickle.load(f1)
    params = {'s': 0.6, 'c': 1, 'h': 0.5, 'target_steps': 100, 'q0': 0.1}
    res = run_model(params)
    wt, mut = res['p'], res['q']
    plt.plot(wt, color = 'orange', label = 'wild-type')
    plt.plot(mut, color = 'blue', label = 'mutant')
    plt.ylabel('Allele Frequency')
    plt.xlabel('Time')
    plt.title('Allele frequency dynamics in non-gene drive population')
    plt.grid(True)
    plt.legend(title='Allele', bbox_to_anchor=(0.8, 0.5), loc='center left')
    plt.show()

def plot_gd(ts, tc):
    colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']
    with open('pickle/allgdres001.pickle', 'rb') as f1:
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


def getHapseMapDiff(currH):
    '''
    Given current h value, plot an Error heatmap for haploid_se vs GD
    '''
    loadFile = f"h{currH}_allgdres001G.pickle" #decides the density of the dots
    gd_results = load_pickle(loadFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = load_pickle(f"h{currH}_gametic_stability_res.pickle")
    # Name output files (txt + pickle)
    is001 = "001" if '001' in loadFile else '01'
    outTXTFile = f"h{currH}_mappingdiff_gdhapse{is001}.txt"
    savedPickle = f"h{currH}_mappingdiff_gdhapse{is001}.pickle"

    gds = [res['q'] for res in gd_res.values()]

    valid_configs = []
    for (s, c, h) in gd_configs:
        if ((s, c, h), 1.0) in stabilityRes['Fixation']: ### NEED TO USE ANALYTICAL BOUNDS
        # if gd_res[(s, c, h)]['state'] == 'fix':
            valid_configs.append((s, c, h))

    saved = dict()

    config_index = {conf: i for i, conf in enumerate(gd_res.keys())}

    for (s, c, h) in valid_configs:

        gd_curve = gd_res[(s,c,h)]['q']

        # plot haploid using Se
        paramSe = {'s':s, 'c':c, 'n': 500, 'h':h, 'target_steps': 40000, 'q0': 0.001}
        hapSe = haploid_se(paramSe)['q']

        curveDiff = euclidean(gd_curve, hapSe)
        saved[(s, c, h)] = curveDiff
        ### DEBUGGING
        # if (math.isclose(s, 0.8) and math.isclose(c,0.9)):
        #     with open("s0.8_c0.9_hapse_diff.txt", 'w') as fout:
        #         fout.write(f"gd: {gd_curve}\nhapse: {hapSe}\nDiff:{curveDiff}")
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(np.arange(len(hapSe)), hapSe, label = 'hapse')
        #     plt.plot(np.arange(len(gd_curve)), gd_curve,label = 'gd')
        #     plt.legend(title='Model', bbox_to_anchor=(1, 1.05), loc='upper left')
        #     plt.show()

    #write the diff to a txt file
    with open(outTXTFile, 'w') as fout:
        fout.write(f"Configuration\tHapse Mapping Diff\n")
        for key, diff in saved.items():
            fout.write(f"{key}\t{diff}\n")   

    save_pickle(savedPickle, saved)

def plotMapDiff(currH):
    gd_results = load_pickle(f"h{currH}_allgdres001G.pickle")
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = load_pickle(f"h{currH}_gametic_stability_res.pickle")
    hapseDiffMap = load_pickle(f"h{currH}_mappingdiff_gdhapse001.pickle")

    valid_configs = []
    for (s, c, h) in gd_configs:
        # if ((s, c, h), 1.0) in stabilityRes['Fixation']: ### NEED TO USE ANALYTICAL BOUNDS
        if gd_res[(s, c, h)]['state'] == 'fix':
            valid_configs.append((s, c, h))

    all_s = sorted(set(s for (s,c,h) in valid_configs))
    all_c = sorted(set(c for (s,c,h) in valid_configs))

    diff_map = np.full((len(all_s), len(all_c)), np.nan)

    for (s, c, h) in valid_configs:
        # The index of this config in wms
        row_idx = all_s.index(s)
        col_idx = all_c.index(c)
        diff_map[row_idx, col_idx] = hapseDiffMap[(s,c,h)]

    X, Y = np.meshgrid(all_c, all_s)
    cmap = plt.get_cmap('BuPu')
    cmap.set_bad('white')

    # Define min and max values for color normalization
    vmin, vmax = min_val, max_val
    vmin2, vmax2 = np.nanmin(diff_map), np.nanmax(diff_map)
    print(vmin2, vmax2)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(X, Y, diff_map, shading='auto', cmap=cmap, norm=norm)
    cbar = plt.colorbar(mesh)
    cbar.set_label('Difference')

    plt.xlabel('c', fontsize = 15)
    plt.ylabel('s', fontsize = 15)
    plt.title(f'Difference Heatmap: GD vs. Haploid Se at h = {currH}')
    plt.show()


'''
Partition plot of final state based on simulation
partition graph: fixed h, x is c, y is s, color indicates s-eff/final state in gd simulation
'''
def partition():
    # loading
    # with open('pickle/sch_to_s_results.pickle', 'rb') as f:
    #     map_results = pickle.load(f)
    gd_results = load_pickle("h05_allgdresG.pickle")
    gd_configs, gd_res = gd_results[0], gd_results[1]

    # seffmap, ngdcurves = map_results['map'], map_results['ngC']
    slist = sorted(set([conf[0] for conf in gd_configs]))
    clist = sorted(set([conf[1] for conf in gd_configs]))
    s_effs = []
    configurations = []
    states = []
    finals = []
    statemap = {'fix': 2.5, 'loss': 0.5, 'stable': 1.5}
    h = 0.5 # need to change this!
    for s in slist:
        for c in clist:
            configurations.append((s, c))
            # s_effs.append(seffmap[(s, c, h)][0])
            # if gd_res[(s,c,h)]['state'] == 'fix': plot_gd(s, c)
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

'''
mapping curves of ngd and gd populations for a single configuration
Need to change the loaded files according to plotted curves
'''
def plot_mapping(currH):
    # loading results
    gd_results = load_pickle(f"h{currH}_allgdres001G.pickle")
    gradResult = load_pickle(f"h{currH}_gradient_G_fix.pickle") # all gradient results
    # gridResult_diploid = load_pickle(f"h{currH}_grid_G_fix.pickle") # all hap grid results
    gridResult = load_pickle(f"h{currH}_hap_grid_G_fix.pickle")

    gd_configs, gd_res = gd_results[0], gd_results[1]

    sMap_grid, wms_grid = gridResult['map'], gridResult['ngC']
    # sMap_grid_diploid, wms_grid_diploid = gridResult_diploid['map'], gridResult_diploid['ngC']
    sMap_grad, wms_grad = gradResult['map'], gradResult['ngC']

    plt.figure(figsize = (15, 7))

    for (s, c, h) in gd_configs:
        if (math.isclose(s, 0.4) and math.isclose(c, 0.6)):
        # if s < c and math.isclose(s, 0.1):
            # plot grid (hap/diploid?)
            if (s, c, h) in sMap_grid and type(sMap_grid[(s,c,h)]) != tuple:
                best_s_grid = sMap_grid[(s, c, h)]
                i = list(sMap_grid.keys()).index((s, c, h))
                wm_curve_grid = wms_grid[i]
                time0 = np.arange(0, len(wm_curve_grid))
                w_color1 = 'y'
                plt.plot(time0, wm_curve_grid, marker = 'o', color = w_color1, markersize=3, linestyle = '-', label = f'grid search haploid NGD (s = {best_s_grid})')
            # if (s, c, h) in sMap_grid_diploid:
            #     print("TRUEE")
            #     best_s_grid, best_h_grid = sMap_grid_diploid[(s, c, h)][0], sMap_grid_diploid[(s, c, h)][1]
            #     w_color0 = 'g'
                # i = list(sMap_grid_diploid.keys()).index((s, c, h))
                # wm_curve_grid = wms_grid_diploid[i]
                # time1 = np.arange(0, len(wm_curve_grid))
                # plt.plot(time1, wm_curve_grid, marker = 'o', color = w_color0, markersize=3, linestyle = '-', label = f'grid search diploid NGD (s = {best_s_grid}, h = {best_h_grid})')

            # PLOT GRADIENT (HAP/DIPLOID)
            # cmap2 = plt.get_cmap('GnBu') # grid mapping curves
            # w_color2 = cmap2(abs(best_s_grad))
            w_color2 = "b"
            if (s, c, h) in sMap_grad and len(sMap_grad[(s,c,h)]) < 2:
                best_s_grad = sMap_grad[(s, c, h)]
                paramHap = {'s':best_s_grad, 'n': 500, 'target_steps': 40000, 'q0': 0.001}
                wm_curve_hap_grad = haploid(paramHap)['q']
                time4 = np.arange(0, len(wm_curve_hap_grad))
                plt.plot(time4, wm_curve_hap_grad, marker = 's', color = w_color2, markersize=3, linestyle = '-', label = f'gradient descent haploid NGD s = {best_s_grad}')
            # best_h_grid = sMap_grid[(s, c, h)][1]
            else:
                best_s_grad, best_h_grad = sMap_grad[(s, c, h)][0], sMap_grad[(s, c, h)][1]
                print("BEFORE", best_s_grad, best_h_grad)
                best_s_grad, best_h_grad = -0.52, 0.22
                wm_curve_grad = wm(best_s_grad, best_h_grad, 40000, 0.001)['q']
                time2 = np.arange(0, len(wm_curve_grad))
                plt.plot(time2, wm_curve_grad, marker = 's', color = w_color2, markersize=3, linestyle = '-', label = f'gradient descent diploid NGD s = {best_s_grad}, h = {best_h_grad}')
            print(best_s_grid)

            # PLOT GD CURVE
            cmap = plt.get_cmap("Reds")
            gd_color = cmap(0.75) # original gene-drive curve

            # PLOT HAPSE
            paramSe = {'s':s, 'c':c, 'n': 500, 'h': h, 'target_steps': 40000, 'q0': 0.001}
            se = h*s-c+c*h*s
            hapSe = haploid_se(paramSe)['q']
            cmapSe = plt.get_cmap("Greens")
            se_color = cmapSe(0.75) # original gene-drive curve

            # nearby = []
            # for hnew in np.arange(best_h_grid-0.2, best_h_grid+0.3, 0.1):
            #     newc = wm(best_s_grid, hnew, params['target_steps'], params['q0'])['q']
            #     nearby.append(newc)
            #     time = np.arange(0, len(newc))
            #     h_color = cmap2(abs(hnew-best_h_grid)*4)
            #     plt.plot(time, newc, marker = 'o', color = h_color, markersize=3, linestyle = '-', label = f"NGD s = {'%.3f' % best_s_grid}, h = {'%.3f' % hnew}")      

            time = np.arange(0, len(gd_res[(s, c, h)]['q']))
            time_se = np.arange(0, len(hapSe))
            time3 = np.arange(0, len(gd_res[(s, c, h)]['q'])-1)
            print("plotting gd and mapped curves")

            plt.plot(time, gd_res[(s, c, h)]['q'], color = gd_color, label = f"s = {s}, c = {c}, h = {h}")
            plt.plot(time_se, hapSe, marker = 'o', color = se_color, markersize=3, label = f"haploid using Se (s = {'%.3f'%se})")
            # plt.plot(time3, gd_res[(s, c, h)]['w_bar'], color = 'b', label = f"wbar for s = {s}, c = {c}, h = {h}")

    plt.ylabel('Gene Drive/Mutant Allele Frequency', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.title(f"Comparison of gene drive and different mapping results at h = {currH}")
    plt.tight_layout(rect=[0, 0, 0.7, 1])
    plt.grid(True)
    legend = plt.legend(title='population condition', bbox_to_anchor=(1, 1.05), loc='upper left')
    export_legend(legend)
    print("showing")
    plt.show()



def ploterror(currH):
    '''
    mappingdiff: dictionary of (s,c,h): error
    '''

    diffmap = load_pickle(f"h{currH}_mappingdiff_gdhapse001.pickle") ###CHANGE NAME IF RUNNING 0.01
    # print(f"in ploterror: {currH}\n{diffmap}")
    stability = load_pickle(f"h{currH}_gametic_stability_res.pickle")

    # with open('pickle/mappingdiff_grid.pickle', 'rb') as f:
    #     diffmap = pickle.load(f)
    
    # print(diffmap)
    ngd_s = sorted(set([conf[0] for conf in diffmap.keys()]))
    ngd_c = sorted(set([conf[1] for conf in diffmap.keys()]))
    # print(ngd_s,ngd_c)
    # print(len(ngd_s), len(ngd_c))
    errors = np.full((len(ngd_s), len(ngd_c)), np.nan)

    for (s, c, h), error in diffmap.items():
        # print(s, c, h)
        if ((s, c, h), 1.0) in stability['Fixation']:
            i = ngd_s.index(s)  # Row index (h axis)
            j = ngd_c.index(c)  # Column index (s axis)
            errors[i, j] = error
    
    # print(errors)
    print(min(diffmap.values()), max(diffmap.values()))

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        errors,
        aspect='auto',
        cmap='BuPu',
        origin='lower',
        extent=[min(ngd_s), max(ngd_s), min(ngd_c), max(ngd_c)],
        vmin=min_val,            # set color scale min
        vmax=max_val            # set color scale max
    )
    plt.colorbar(label='Mean Squared Error (Error of mapping)')
    plt.xlabel('c in gene drive', fontsize = 15)
    plt.ylabel('s in gene drive', fontsize = 15)
    plt.title(f"Mapping Error from Gene-Drive to Non-Gene-Drive Haploid Model at h = {currH} with Grid Search")
    plt.show()


def gd_to_ngd_diff():
    gd_s = 0.6
    gd_c = 0.4
    gd_h = 0.0
    # mappingdiff: (s,c,h): error of map
    with open('pickle/mappingdiff.pickle', 'rb') as f:
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
    with open('pickle/sch_to_s_results.pickle', 'rb') as f:
        map_results = pickle.load(f)
    
    with open('pickle/gd_simresults.pickle', 'rb') as f2:
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