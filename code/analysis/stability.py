import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import sys, os

import sympy as sp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import wm

def compute_lambda_gd():
# Define symbols
    s, c, h, q = sp.symbols('s c h q')

    # Define numerator and denominator
    sc = (1 - h*s) * c
    sn = 0.5 * (1 - h * s) * (1 - c)
    numerator = q**2 * (1 - s) + 2 * q * (1 - q) * c * (sc+sn)
    wbar = q**2 * (1 - s) + 2 * q * (1 - q) * (sc+2*sn) + (1 - q)**2

    # Define q(t+1)
    q_next = numerator / wbar

    # Compute the derivative ∂q(t+1)/∂q(t)
    dq_next_dq = sp.simplify(sp.diff(q_next, q))

    dq_dq_func = sp.lambdify((s, c, h, q), dq_next_dq, "numpy")

    # Print the simplified result
    sp.pprint(dq_next_dq)
    return dq_dq_func

def get_gd_stability(s, c, h, q, dfunc):
    try:
        slope = dfunc(s, c, h, q)
    except ZeroDivisionError:
        slope = 'NA'
        print(f"Division by zero: config = {(s, h, q)}")
    return slope

def get_ngd_stability_old(s, h, q):
    num = 2*q**2*(1-q)*(1-s)*(1-h*s) - q**2*(1-2*q)*(1-s)*(1-h*s) + 2*q*(1-q)**2*(2-s-h*s) + (1-2*q)*(1-q)**2*(1-h*s)

    denom = (q**2*(1-s) + 2*(1-q)*q*(1-h*s) + (1-q)**2)**2

    if denom != 0:
        slope = num/denom
    else:
        slope = 'NA'
        print(f"no slope result, config = {(s, h, q)}")
    
    return slope

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_pickle,save_pickle

def derivative(params):
    s, c, h = params['config']
    q = params['currq']
    if params['conversion'] == 'zygotic':
        sc = c*(1-s)
    else:
        sc = c*(1-h*s)
    sn = 0.5*(1-c)*(1-h*s)

    num = 2*q**2*(1-q)*(1-s)*(2*sn+1) - 2*q**2*(1-2*q)*(1-s)*sn + 2*q*(q-1)**2*(1-s+2*sn+2*sc) + 2*(1-2*q)*(1-q)**2*(sn+sc)

    denom = (q**2*(1-s) + 2*(1-q)*q*(2*sn+sc) + (1-q)**2)**2

    if denom != 0:
        slope = num/denom
    else:
        slope = 'NA'
        print(f"no slope result, config = {params['config']}")
    
    return slope

def get_eq(params):
    s, c, h = params['config']
    q = params['q0']
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
get NGD phase partition
'''

# get_eq({'config':(0.2, 0.8, 0), 'q0':0.001, 'conversion': "zygotic"})
'''
return allres: dict{"state": [(config1, eq), ...]}
'''
def runall(params):
    allres = {'Stable': [], 'Unstable': [], 'dq=1': [], 'Fixation': [], 'Loss':[]}
    if params['conversion'] == 'zygotic':
        gdres_file = f"gd_simulation_results/h{params['config'][2]}_allgdres001.pickle"
    else:
        gdres_file = f"gd_simulation_results/h{params['config'][2]}_allgdres001G.pickle"
    g_res = load_pickle(gdres_file)
    configs, res = g_res[0], g_res[1]
    print('allconfigs', configs[:2])

    for s in np.arange(0.01, 1.01, 0.01):
        for c in np.arange(0.01, 1.01, 0.01):
            h = params['config'][2]
            params['config'] = (round(float(s), 3),round(float(c), 3),round(float(h), 3))

            alleq=get_eq(params) ### get all eq points for the current config
            eq1, eq2, eq3 = alleq['q1'], alleq['q2'], alleq['q3']

            # compute stability for eq3

            if eq3 != 'NA' and 0 < eq3 and eq3 < 1 and not math.isclose(eq3, 0) and not math.isclose(eq3, 1):
                params['currq']=eq3
                eq_stability = derivative(params)
                # print(f"config = {params['config']}, slope={eq_stability}")

                # if slope < 1: stable
                if 0 < eq3 < 1 and eq_stability != 'NA':
                    if abs(eq_stability) < 1:
                        allres['Stable'].append((params['config'], eq3))
                    elif abs(eq_stability) > 1:
                        allres['Unstable'].append((params['config'], eq3))
                    elif math.isclose(eq_stability, 1):
                        allres['dq=1'].append((params['config'], eq3))
                # find fixation and loss
            elif eq3 != 'NA' and (eq3 >= 1 or eq3 <= 0 or math.isclose(eq3, 0) or math.isclose(eq3, 1)):
                if math.isclose(eq3, 1) or res[params['config']]['state'] == 'fix':
                    allres['Fixation'].append((params['config'], 1.0))
                elif res[params['config']]['state'] == 'loss':
                    allres['Loss'].append((params['config'], 0.0))
            # if math.isclose(s, 0.08) and math.isclose(c, 0.08):
            #     print('something eq3', eq3, math.isclose(eq3, 1))
            #     print(((s, c, h), eq3) in allres['Fixation'])
            elif eq3 == 'NA':
                if res[params['config']]['state'] == 'loss':
                    allres['Loss'].append((params['config'], 0.0))
                elif res[params['config']]['state'] == 'fix':
                    allres['Fixation'].append((params['config'], 1.0))

        
    return allres
            
def plot_all(allres, params):
    fig, ax = plt.subplots()
    for state in allres:

        unstable_configurations = allres[state]
        s_values = [config[0][0] for config in unstable_configurations]  # First element in the tuple (s)
        c_values = [config[0][1] for config in unstable_configurations]  # Second element in the tuple (c)
        eq_values = [max(0, min(1, config[1])) for config in unstable_configurations]
        # print(state, eq_values)

        
        if eq_values != []:
            # Plotting
            norm = mcolors.Normalize(0.0, 1.0)
            # if state == 'Stable' or state == 'Unstable':
            cmap = plt.cm.get_cmap('viridis')
            # plt.figure(figsize=(8, 6))
            # plt.scatter(c_values, s_values, alpha=0.7, edgecolors='k', label=f"Unstable Configurations" )
            ax.scatter(c_values, s_values, c=eq_values, cmap=cmap, s=50, norm=norm)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Gene Drive Allele Frequency at Eq')

    # Adding labels, title, and legend
    ax.set_xlabel('Conversion Factor (c)', fontsize=12)
    ax.set_ylabel('Selection Coefficient (s)', fontsize=12)
    # plt.title(f'Scatter Plot of {state} Configurations', fontsize=14)
    ax.set_title(f"Partition of Gene Drive Configurations Based on Stability (H={params['config'][2]})", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    # plt.legend(fontsize=10)

    # Show plot
    plt.grid(True)
    plt.show()


def plot_regions(allres, state, conversion):
    unstable_configurations = allres[state]
    s_values = [config[0][0] for config in unstable_configurations]  # First element in the tuple (s)
    c_values = [config[0][1] for config in unstable_configurations]  # Second element in the tuple (c)
    h_val = unstable_configurations[0][0][2]
    eq_values = [max(0, min(1, config[1])) for config in unstable_configurations]

    # Plotting
    fig, ax = plt.subplots()
    norm = mcolors.Normalize(min(eq_values), max(eq_values))
    cmap = plt.cm.get_cmap('viridis')
    # plt.figure(figsize=(8, 6))
    # plt.scatter(c_values, s_values, alpha=0.7, edgecolors='k', label=f"Unstable Configurations" )
    scatter = ax.scatter(c_values, s_values, c=eq_values, cmap=cmap, s=50, norm=norm)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Gene Drive Allele Frequency at Eq')

    # Adding labels, title, and legend
    ax.set_xlabel('Conversion Factor (c)', fontsize=12)
    ax.set_ylabel('Selection Coefficient (s)', fontsize=12)
    # plt.title(f'Scatter Plot of {state} Configurations', fontsize=14)
    ax.set_title(f"Plot of Equilibrium Values in the {state} Regime ({conversion} model)", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    # plt.legend(fontsize=10)

    # Show plot
    plt.grid(True)
    plt.show()

#######################################################################################
# NGD stability 
#######################################################################################
def compute_lambda():
# Define symbols
    q, s, h = sp.symbols('q s h')

    # Define numerator and denominator
    numerator = q**2 * (1 - s) + q * (1 - q) * (1 - h * s)
    wbar = q**2 * (1 - s) + 2 * q * (1 - q) * (1 - h * s) + (1 - q)**2

    # Define q(t+1)
    q_next = numerator / wbar

    # Compute the derivative ∂q(t+1)/∂q(t)
    dq_next_dq = sp.simplify(sp.diff(q_next, q))

    dq_dq_func = sp.lambdify((s, h, q), dq_next_dq, "numpy")

    # Print the simplified result
    sp.pprint(dq_next_dq)
    return dq_dq_func

dfunc_ngd = compute_lambda()

def get_ngd_stability(s, h, q, dfunc_ngd):
    try:
        slope = dfunc_ngd(s, h, q)
    except ZeroDivisionError:
        slope = 'NA'
        print(f"Division by zero: config = {(s, h, q)}")
    return slope

'''
Given the NGD parameter (s, h), return the stable eq
'''
def get_eq_ngd(s, h):
    if not math.isclose(h, 0.5):
        if (math.isclose(2*h*s - s, 0.0)):
            print(f'Division by zero: config = {(s, h)}')
            return 'NA'
        return (h*s)/(2*h*s - s)
    return 'NA'

print(get_eq_ngd(-0.2, 0.1))


def runall_ngd(q_init):
    allres = {'Stable': [], 'Unstable': [], 'dq=1': [], 'Fixation': [], 'Loss': []}
    dfunc = compute_lambda()

    for s in np.arange(-2, 2, 0.1):
        for h in np.arange(-2, 2, 0.1):
            s = round(float(s), 3)
            h = round(float(h), 3)
            # print(s, h)

            # eq = get_eq_ngd(s, h)
            eq = wm(s, h, 40000, q_init)['q'][-1]

            if eq != 'NA':
                # print("getting stability")
                # print(s, h, eq)
                slope = get_ngd_stability(s, h, eq, dfunc)

                if eq >= 1 or math.isclose(eq, 1.0):
                    allres['Fixation'].append(((s, h), 1.0))
                elif eq <= 0 or math.isclose(eq, 0.0):
                    allres['Loss'].append(((s, h), 0.0))
                else:
                    if slope != 'NA':
                        if abs(slope) < 1:
                            allres['Stable'].append(((s, h), eq, slope))
                        elif abs(slope) > 1:
                            allres['Unstable'].append(((s, h), eq, slope))
                        elif math.isclose(slope, 1.0):
                            allres['dq=1'].append(((s, h), eq))
            else:
                print('NO EQ:', s, h)
                res = wm(s, h, 40000, q_init)
                res_state = res['state'] 
                if res_state == 'loss':
                    allres['Loss'].append(((s, h), 0.0))
                elif res_state == 'fix':
                    allres['Fixation'].append(((s, h), 1.0))
                else:
                    print(f"Skipping invalid config s={s}, h={h}, final={res['q'][-1]}")

    with open(f"stability_res/NGD_partition_sim_q{q_init}.txt", "w") as out:
        for state in allres:
            if state != "Stable" and state != "Unstable":
                for config, eq in allres[state]:
                    out.write(f"{state}: s={config[0]:.3f}, h={config[1]:.3f}, eq={eq:.4f}\n")
            else:
                for config, eq, slope in allres[state]:
                    out.write(f"{state}: s={config[0]:.3f}, h={config[1]:.3f}, eq={eq:.4f}, slope = {slope:.5f}\n")
    return allres


def plot_all_ngd(allres, q_init):
    fig, ax = plt.subplots()
    # state = 'Unstable'
    for state in allres.keys():
        unstable_configurations = allres[state]
        s_values = [config[0][0] for config in unstable_configurations]
        h_values = [config[0][1] for config in unstable_configurations]
        eq_values = [max(0, min(1, config[1])) for config in unstable_configurations]

        if eq_values:
            norm = mcolors.Normalize(0.0, 1.0)
            cmap = plt.cm.get_cmap('viridis')
            ax.scatter(s_values, h_values, c=eq_values, cmap=cmap, s=50, norm=norm)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Allele Frequency at Eq')

    ax.set_xlabel('Selection Coefficient (s)', fontsize=12)
    ax.set_ylabel('Dominance (h)', fontsize=12)
    ax.set_title(f"NGD Final Phase Partition (q_init={q_init})", fontsize=14)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.savefig(f"plot_partition/NGD_partition_sim_q{q_init}.jpg", dpi=600)
    plt.close()


def main():
    s=0.6
    c=0.6
    h=0.8
    q_init=0.001
    regime = 'Fixation'
    params = {'config':(s, c, h), 'q0':q_init, 'conversion': "gametic"}
    filename = f"h{h}_{params['conversion']}_stability_res.pickle"

    # run through all configurations
    # allres = runall(params)
    # save_pickle(filename, allres) ### change file name
    
    # allres = load_pickle(filename)

    # print(allres)
    # f_out = open(f"stability_res/h{h}_g_{regime}.txt", 'w') #### change file name!!!!
    # f_out.write(f"gene drive model configuration\t\tequilibrium\n")
    # for state in allres.keys():
    #     if state == regime:  ### change state check
    #         for config, eq_val in allres[state]:
    #             f_out.write(f"(s, c, h) = {config}\t\teq = {eq_val}\n")
    # plot_regions(allres, regime, params['conversion'])
    

    ### PLOT NGD PARTITION
    q_init = 0.001
    file = f"ngd_sim_stability_q{q_init}.pickle"
    allres_ngd = runall_ngd(q_init)
    save_pickle(file, allres_ngd)

    allres_ngd = load_pickle(file)

    plot_all_ngd(allres_ngd, q_init)
    # plot_all(allres, params)
    # dfunc = compute_lambda()
    # params = {'config': (0.5,0.3,0.3), 'q0': 0.001, 'conversion': 'gametic'}
    # eq = get_eq(params)['q3']
    # h_ngd = eq/(2*eq-1)
    # for s_ngd in np.arange(-2, 0, 0.1):
    #     ngd_eq = wm(s_ngd, h_ngd, 40000, 0.001)['q'][-1]
    #     ngd_stat = get_ngd_stability(s_ngd, h_ngd, eq, dfunc)
    #     print("######### GD EQ:", eq)
    #     print("========= NGD EQ:", ngd_eq)
    #     print("++++++++++++++++++++++")
    #     print(f"s_ngd = {s_ngd}, dq = {ngd_stat}")
    #     print("----------------------")
    #     ngd_stat_old = get_ngd_stability_old(s_ngd, h_ngd, eq)
    #     print(f"s_ngd = {s_ngd}, dq_old = {ngd_stat_old}")



if __name__ == '__main__':
    main()

