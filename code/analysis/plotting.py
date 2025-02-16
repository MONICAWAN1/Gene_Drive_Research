import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import math
import pickle

from models import run_model, wm, haploid_se
from utils import export_legend, euclidean, load_pickle, save_pickle

colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']

diffmap1 = load_pickle("mappingdiff.pickle")
diffmap2 = load_pickle("mappingdiff_grid.pickle")
diffmap3 = load_pickle("mappingdiff_grid_hap.pickle")
diffmap5 = load_pickle("mappingdiff_gdhapse.pickle")
diffmap6 = load_pickle("mappingdiff_gdhapse_h05.pickle")
diffmap7 = load_pickle("h05_mappingdiff_grid_hap.pickle")
# print(diffmap1, '\n', diffmap2)

all_values = list(diffmap3.values()) + list(diffmap5.values())
min_val = min(all_values)
max_val = max(all_values)
print(min_val, max_val)

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


def plotMapDiff():
    gd_results = load_pickle("h05_allgdres001G.pickle")
    gd_configs, gd_res = gd_results[0], gd_results[1]
    # with open('pickle/hap_grid_gametic_fix_results.pickle', 'rb') as f:
    #     mapResult = pickle.load(f)

    # the hap_se results
    mapResult = load_pickle("allhapseres001G.pickle")

    # print(mapResult)
    # sMap, wms = mapResult['map'], mapResult['q']
    gds = [res['q'] for res in gd_res.values()]

    valid_configs = []
    for (s, c, h) in gd_configs:
        if gd_res[(s,c,h)]['state']=='fix':
            valid_configs.append((s, c, h))

    all_s = sorted(set(s for (s,c,h) in valid_configs))
    all_c = sorted(set(c for (s,c,h) in valid_configs))

    diff_map = np.full((len(all_s), len(all_c)), np.nan)
    saved = dict()

    config_index = {conf: i for i, conf in enumerate(gd_res.keys())}

    for (s, c, h) in valid_configs:
        # The index of this config in wms
        idx = config_index[(s, c, h)]
        gd_curve = gds[idx]

        # plot haploid using Se
        paramSe = {'s':s, 'c':c, 'n': 500, 'h': 0, 'target_steps': 40000, 'q0': 0.001}
        hapSe = haploid_se(paramSe)['q']

        curveDiff = euclidean(gd_curve, hapSe)/min(len(hapSe), len(gd_curve))
        row_idx = all_s.index(s)
        col_idx = all_c.index(c)
        diff_map[row_idx, col_idx] = curveDiff
        saved[(s, c, h)] = curveDiff

    # save_pickle("mappingdiff_gdhapse_h05.pickle", saved)

    X, Y = np.meshgrid(all_c, all_s)
    cmap = plt.get_cmap('YlOrBr')
    cmap.set_bad('white')

    # Define min and max values for color normalization
    vmin, vmax = min_val, max_val
    # vmin, vmax = np.nanmin(diff_map), np.nanmax(diff_map)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(X, Y, diff_map, shading='auto', cmap=cmap, norm=norm)
    cbar = plt.colorbar(mesh)
    cbar.set_label('Difference (mapped vs. hapSe)')

    plt.xlabel('c')
    plt.ylabel('s')
    plt.title('Difference Heatmap: GD vs. Haploid Se at h = 0.5')
    plt.show()


# sweep through all configurations, get final state
# partition graph (fixed h, x is c, y is s, color indicates s-eff/final state in gd simulation)
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

# mapping curves of ngd and gd populations ###########
def plot_mapping():
    # loading results
    gd_results = load_pickle("allgdres001G.pickle")
    gradResult = load_pickle("grad_gametic_fix_results.pickle") # all gradient results
    gridResult = load_pickle("hap_grid_gametic_fix.pickle") # all gradient results


    gd_configs, gd_res = gd_results[0], gd_results[1]

    sMap_grid, wms_grid = gridResult['map'], gridResult['ngC']
    sMap_grad, wms_grad = gradResult['map'], gradResult['ngC']

    gd_configs = [key for key in sMap_grid]
    # print(gd_configs)

    plt.figure(figsize = (15, 7))

    for i in range(len(gd_configs)):
        s, c, h = gd_configs[i]
        if (math.isclose(s, 0.2) and math.isclose(c, 0.8)):
        # if s < c and math.isclose(s, 0.1):
            best_s_grid = sMap_grid[(s, c, h)]
            # best_h_grid = sMap_grid[(s, c, h)][1]
            # best_s_grad, best_h_grad = sMap_grad[(s, c, h)][0], sMap_grad[(s, c, h)][1]
            # print(best_s_grid)

            # cmap = plt.get_cmap(colormaps[i % len(colormaps)])
            cmap = plt.get_cmap("Reds")
            gd_color = cmap(c) # original gene-drive curve
            cmap1 = plt.get_cmap('PuRd')
            w_color1 = cmap1(abs(best_s_grid)) # gradient mapping curves

            # cmap2 = plt.get_cmap('GnBu') # grid mapping curves
            # w_color2 = cmap2(abs(best_s_grad))

            paramSe = {'s':s, 'c':c, 'n': 500, 'h': 0, 'target_steps': 40000, 'q0': 0.001}
            hapSe = haploid_se(paramSe)['q']

            cmapSe = plt.get_cmap("GnBu")
            se_color = cmapSe(c) # original gene-drive curve

            # nearby = []
            # for hnew in np.arange(best_h_grid-0.2, best_h_grid+0.3, 0.1):
            #     newc = wm(best_s_grid, hnew, params['target_steps'], params['q0'])['q']
            #     nearby.append(newc)
            #     time = np.arange(0, len(newc))
            #     h_color = cmap2(abs(hnew-best_h_grid)*4)
            #     plt.plot(time, newc, marker = 'o', color = h_color, markersize=3, linestyle = '-', label = f"NGD s = {'%.3f' % best_s_grid}, h = {'%.3f' % hnew}")      

            # wm_curve_grid = wms_grid[i]
            # wm_curve_grad = wms_grad[i]
            # print(wm_curve_grid)
            # print(wm_curve)
            # time1 = np.arange(0, len(wm_curve_grid))
            # time2 = np.arange(0, len(wm_curve_grad))
            time = np.arange(0, len(gd_res[(s, c, h)]['q']))
            time_se = np.arange(0, len(hapSe))
            time3 = np.arange(0, len(gd_res[(s, c, h)]['q'])-1)
            print("plotting gd and mapped curves")
            # plt.plot(time1, wm_curve_grid, marker = 'o', color = w_color1, markersize=3, linestyle = '-', label = f'grid mapped haploid NGD s = {best_s_grid}')
            # plt.plot(time2, wm_curve_grad, marker = 's', color = w_color2, markersize=3, linestyle = '-', label = f'gradient mapped NGD s = {best_s_grad}, h = {best_h_grad}')
            plt.plot(time, gd_res[(s, c, h)]['q'], color = gd_color, label = f"s = {s}, c = {c}, h = {h}")
            plt.plot(time_se, hapSe, marker = 'o', color = se_color, markersize=3, label = f"haploid using Se")
            # plt.plot(time3, gd_res[(s, c, h)]['w_bar'], color = 'b', label = f"wbar for s = {s}, c = {c}, h = {h}")

    # param_text = f"orange: non-gene-drive population\nblue: population with gene drive\nq_initial = {params['q0']}"
    # plt.figtext(0.6, 0.2, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    plt.ylabel('Gene Drive/Mutant Allele Frequency', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.title("Comparison of gene drive and haploid Se results at h = 0.5")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.grid(True)
    legend = plt.legend(title='population condition', bbox_to_anchor=(1, 1.05), loc='upper left')
    export_legend(legend)
    print("showing")
    plt.show()



def ploterror():
    # mappingdiff: (s,c,h): error of map

    diffmap = load_pickle("h05_mappingdiff_grid_hap.pickle")

    # with open('pickle/mappingdiff_grid.pickle', 'rb') as f:
    #     diffmap = pickle.load(f)
    
    # print(diffmap)
    ngd_s = sorted(set([conf[0] for conf in diffmap.keys()]))
    ngd_c = sorted(set([conf[1] for conf in diffmap.keys()]))
    print(ngd_s,ngd_c)
    print(len(ngd_s), len(ngd_c))
    errors = np.zeros((len(ngd_s), len(ngd_c)))

    for (s, c, h), error in diffmap.items():
        # print(s, c, h)
        if s > 0:
            i = ngd_s.index(s)  # Row index (h axis)
            j = ngd_c.index(c)  # Column index (s axis)
            errors[i, j] = error
    
    # print(errors)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        errors,
        aspect='auto',
        cmap='BuPu',
        origin='lower',
        extent=[min(ngd_s), max(ngd_s), min(ngd_c), max(ngd_c)],
        vmin=min_val,            # set color scale min
        vmax=max_val             # set color scale max
    )
    plt.colorbar(label='Euclidean Distance (Error of mapping)')
    plt.xlabel('c in gene drive')
    plt.ylabel('s in gene drive')
    plt.title(f"Error Heatmap for the Grid Mapping from Gene-Drive to Non-Gene-Drive Haploid Model at h = 0.5")
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