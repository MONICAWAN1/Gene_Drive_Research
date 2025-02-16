import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import pickle

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
    print(q3)
    return eqs

# get_eq({'config':(0.2, 0.8, 0), 'q0':0.001, 'conversion': "zygotic"})

def runall(params):
    allres = {'eq3': {'Stable': [], 'Unstable': [], 'dq=1': [], 'Fixation': [], 'Loss':[]}}
    if params['conversion'] == 'zygotic':
        with open('pickle/allgdres001.pickle', 'rb') as f1:
            z_res = pickle.load(f1)
            configs, res = z_res[0], z_res[1]
    else:
        with open('pickle/allgdres001gametic.pickle', 'rb') as f1:
            g_res = pickle.load(f1)
            configs, res = g_res[0], g_res[1]

    for s in np.arange(0.01, 1, 0.01):
        for c in np.arange(0.01, 1, 0.01):
            h = 0.5
            params['config'] = (round(float(s), 3),round(float(c), 3),round(float(h), 3))

            alleq=get_eq(params) ### get all eq points for the current config
            eq1, eq2, eq3 = alleq['q1'], alleq['q2'],alleq['q3']
            

            # compute stability for eq3
            for eq in [eq3]:
                if eq != 'NA':
                    params['currq']=eq
                    eq_stability = derivative(params)
                    print(f"config = {params['config']}, slope={eq_stability}")

                    # if slope < 1: stable
                    if 0 < eq < 1 and eq_stability != 'NA':
                        if abs(eq_stability) < 1:
                            allres['eq3']['Stable'].append((params['config'], eq))
                        elif abs(eq_stability) > 1:
                            allres['eq3']['Unstable'].append((params['config'], eq))
                        elif math.isclose(eq_stability, 1):
                            allres['eq3']['dq=1'].append((params['config'], eq))
                    # find fixation and loss
                    # elif eq > 1 or eq < 0:
                    #     if res[params['config']]['state'] == 'fix':
                    #         allres['eq3']['Fixation'].append((params['config'], 1.0))
                    #     elif res[params['config']]['state'] == 'loss':
                    #         allres['eq3']['Loss'].append((params['config'], 0.0))

        
    return allres
            
def plot_all(allres, conversion):
    for eq in allres:
        # if eq == 'eq1':
        #     loss_configurations = allres[eq]["Loss"]
        #     s_values_loss = [config[0][0] for config in unstable_configurations]  # First element in the tuple (s)
        #     c_values_loss = [config[0][1] for config in unstable_configurations]  # Second element in the tuple (c)
        #     eq_values_loss = [config[1] for config in unstable_configurations]
        #     if eq_values_loss != []:
        #     # Plotting
        #         fig, ax = plt.subplots()
        #         norm = mcolors.Normalize(min(eq_values), max(eq_values))
        #         cmap = plt.cm.get_cmap('viridis')
        #         scatter = ax.scatter(c_values_loss, s_values_loss, c=eq_values_loss, cmap=cmap, s=50, norm=norm)
        # if eq == 'eq2':
        #     fix_configurations = allres[eq]["Loss"]
        #     s_values_fix = [config[0][0] for config in unstable_configurations]  # First element in the tuple (s)
        #     c_values_fix = [config[0][1] for config in unstable_configurations]  # Second element in the tuple (c)
        #     eq_values_fix = [config[1] for config in unstable_configurations]
        #     if eq_values_fix != []:
        #         # Plotting
        #         fig, ax = plt.subplots()
        #         norm = mcolors.Normalize(min(eq_values), max(eq_values))
        #         cmap = plt.cm.get_cmap('viridis')
        #         scatter = ax.scatter(c_values_fix, s_values_fix, c=eq_values_fix, cmap=cmap, s=50, norm=norm)
        
        if eq == 'eq3':
            fig, ax = plt.subplots()
            for state in allres[eq]:

                unstable_configurations = allres[eq][state]
                s_values = [config[0][0] for config in unstable_configurations]  # First element in the tuple (s)
                c_values = [config[0][1] for config in unstable_configurations]  # Second element in the tuple (c)
                eq_values = [config[1] for config in unstable_configurations]
                print(state, eq_values)
                
                if eq_values != []:

                    # Plotting
                    norm = mcolors.Normalize(0.0, 1.0)
                    if state == 'Stable' or state == 'Unstable':
                        cmap = plt.cm.get_cmap('viridis')
                    else:
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
    ax.set_title(f"Partition of Gene Drive Configurations Based on Stability (H=0)", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    # plt.legend(fontsize=10)

    # Show plot
    plt.grid(True)
    plt.show()


def plot_regions(allres, eq, state, conversion):
    unstable_configurations = allres[eq][state]
    s_values = [config[0][0] for config in unstable_configurations]  # First element in the tuple (s)
    c_values = [config[0][1] for config in unstable_configurations]  # Second element in the tuple (c)
    eq_values = [config[1] for config in unstable_configurations]

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

def main():
    s=0.6
    c=0.6
    h=0
    q_init=0.001
    params = {'config':(s, c, h), 'q0':q_init, 'conversion': "gametic"}
    filename = f"h0.5_{params['conversion']}_stability_res.pickle"

    # run through all configurations
    allres = runall(params)
    with open(f'pickle/{filename}', 'wb') as f1: ### change file name
        pickle.dump(allres, f1)
    
    with open(f'pickle/{filename}', 'rb') as f1:
        allres = pickle.load(f1)

    print(allres)
    f_out = open('outputs/h0.5_g_unstable.txt', 'w') #### change file name!!!!
    f_out.write(f"gene drive model configuration\t\tequilibrium\n")
    for eq in allres.keys():
        for state in allres[eq].keys():
            if state == 'Unstable':  ### change state check
                for config, eq_val in allres[eq][state]:
                    f_out.write(f"(s, c, h) = {config}\t\teq = {eq_val}\n")
        plot_regions(allres, eq, "Unstable", params['conversion'])
    
    # plot_all(allres, params['conversion'])


if __name__ == '__main__':
    main()

