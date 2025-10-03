# compare the gene drive and mapped non-gene drive simulations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import math
import pandas as pd

from gd_model import gd_simulation
from gd_model_old import run_model
from ngd_model import ngd_simulation
from matrix import find_mapped, jac_and_eigs, get_analytics
from gd_partition import *
from helpers import load_pickle, save_pickle, euclidean
import pickle
from ngd_model_old import wm, haploid

def check_gs(params, m_val, alpha_val):
    s_val = params['s']
    if params['type'] == "haploid":
        return m_val / (1-s_val) + m_val / (1+alpha_val*s_val) >= 1
    else:
        h_val = params['h']
        return jac_and_eigs(s_val, h_val, m_val, alpha_val) < 1

def check_range(params, m, alpha):
    '''
    Check if alpha and m are in the valid range to ensure that 0 <= q <= 1
    '''
    s, q = params['s'], params['q1']
    # check m range
    q_m =  q * m + (1-m) * (1-q) 
    if 1 < q_m or q_m < 0:
        return False
    
    # check alpha range
    if params['type'] == "haploid":
        if 1 - alpha * s <= 0:
            return False
    elif params['type'] == "diploid":
        h = params['h']
        if 1 - h * alpha*s <= 0 or 1 - alpha * s <= 0:
            return False
    return True

def find_m_alpha(params):
    '''
    Find the m and alpha with gene swamping for the haploid model
    '''
    m_range, alpha_range = (0.01, 1.0), (0.01, 2.0) 
    resolution = 500
    m_vals = np.linspace(m_range[0], m_range[1], resolution)
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], resolution)
    for m_val in m_vals:
        for alpha_val in alpha_vals:
            if check_gs(params, m_val, alpha_val) and check_range(params, m_val, alpha_val):
                print(f"Running with params: m={m_val:.3f}, alpha={alpha_val:.3f}")
                return round(m_val, 3), round(alpha_val, 3)

def plot_ngd_single(ngd_params):
    ngd_params['m'] = 0.8
    ngd_params['alpha'] = -0.9
    ngd_result = ngd_simulation(ngd_params)
    plt.figure(figsize=(8,6))
    plt.plot(ngd_result['q1'], label=f'Favoured Allele (q1), m={ngd_params['m']}, alpha={ngd_params['alpha']}')
    plt.plot(ngd_result['q2'], label='Other Allele (q2)')
    paramstr = f"s1={ngd_params["s"]}, s2={ngd_params["s"]*ngd_params['alpha']:.3f}, h={ngd_params["h"]}" if ngd_params['type'] == "diploid" else f"s1={ngd_params["s"]}, s2={ngd_params["s"]*ngd_params['alpha']:.3f}"
    plt.title(f'Non-gene drive {ngd_params['type']} simulation ({paramstr})')
    plt.xlabel('Generations')
    plt.ylabel('Allele Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_gd_all_configs(params):
    configs = []
    s_vals = np.arange(0, 1.01, 0.01)
    c_vals = np.arange(0, 1.01, 0.01)
    h_vals = np.arange(0, 1.01, 0.1)

    # get a list of configurations for simulations 
    for s in s_vals:
        for c in c_vals:
            for h in h_vals:
                configs.append((round(float(s), 3), round(float(c), 3), round(float(h), 3)))

    all_gs_configs = dict()

    for (s, c, h) in configs:
        # print('gd:', (s, c, h))
        params['s'], params['c'], params['h'] = s, c, h
        swamp_configs = run_gd_all(params)
        if swamp_configs != None:
            valid_params = swamp_configs
            all_gs_configs[(s, c, h)] = valid_params
            break
    print(all_gs_configs)
    return all_gs_configs

def run_gd_all(gd_params, tol=1e-6):
    '''
    For a fixed (s, c, h) config, run GD 2 deme simulation across different m and alpha values
    to see what's the maximum possible deme 2 allele frequency
    '''
    gd_params['type'] = "diploid"
    m_range, alpha_range = (0.01, 1.0), (-0.01, -2.0) 
    resolution = 100
    m_vals = np.linspace(m_range[0], m_range[1], resolution)
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], resolution)
    gs_mesh = np.zeros((len(m_vals), len(alpha_vals)))
    for i, m_val in enumerate(m_vals):
        for j, alpha_val in enumerate(alpha_vals):
            if not check_range(gd_params, m_val, alpha_val):
                continue
            gd_params['m'] = m_val
            gd_params['alpha'] = alpha_val
            gd_out = gd_simulation(gd_params)
            # plot_ngd_single(ngd_params, ngd_result)
            if abs(gd_out['q1'][-1]) < tol and abs(gd_out['q2'][-1]) < tol:
                return (m_val, alpha_val)
                
    return None
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("Blues")
    im=plt.imshow(gs_mesh, extent=(alpha_range[0], alpha_range[1], m_range[0], m_range[1]),
               aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(im, label='Final Allele Frequency in deme 2', format='%.1f')
    plt.xlabel('Alpha (Ratio of Relative Fitness)')
    plt.ylabel('Migration Rate (m)')
    plt.title('Heatmap of Final q1 in deme 2 of Non-gene Drive Simulation')
    plt.savefig(f'{gd_params['type']}_ngd_finalq_heatmap_s{gd_params['s']}.png', dpi=600)
    plt.show()

def run_ngd_all(params):
    '''
    Run ngd 2-deme simulation at a fixed s1 value to see what's 
    the maximum possible allele frequency in deme 2 at different m and alpha.
    '''
    ngd_params = params.copy()
    ngd_params['s'] = -ngd_params['s']
    ngd_params['type'] = "haploid"
    m_range, alpha_range = (0.01, 1.0), (-0.01, -2.0)
    resolution = 100
    m_vals = np.linspace(m_range[0], m_range[1], resolution)
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], resolution)

    # Initialize with NaN to differentiate invalid entries
    gs_mesh = np.full((len(m_vals), len(alpha_vals)), np.nan)

    for i, m_val in enumerate(m_vals):
        for j, alpha_val in enumerate(alpha_vals):
            if not check_range(ngd_params, m_val, alpha_val):
                continue
            ngd_params['m'] = m_val
            ngd_params['alpha'] = alpha_val
            ngd_result = ngd_simulation(ngd_params)
            if abs(ngd_result['q2'][-1]) < 0.0001:  # Gene swamping condition
                gs_mesh[i, j] = max(ngd_result['q2'])

    np.savetxt(f"gs_mesh_{ngd_params['type']}_s{ngd_params['s']}.txt", gs_mesh, fmt="%.6f")
    gs_mesh = np.loadtxt(f"gs_mesh_{ngd_params['type']}_s{ngd_params['s']}.txt")

    plt.figure(figsize=(8, 6))

    # Use a colormap that displays NaNs in a distinguishable color
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color='lightgray')  # Color for NaN entries

    # Mask NaNs for imshow
    im = plt.imshow(
        np.ma.masked_invalid(gs_mesh),
        extent=(alpha_range[0], alpha_range[1], m_range[0], m_range[1]),
        aspect='auto',
        origin='lower',
        cmap=cmap
    )

    plt.colorbar(im, label='Final Allele Frequency in deme 2', format='%.1f')
    plt.xlabel('Alpha (Ratio of Relative Fitness)')
    plt.ylabel('Migration Rate (m)')
    plt.title(f'Heatmap of max q2 in deme 2 (s={ngd_params["s"]})')
    plt.savefig(f'{ngd_params["type"]}_ngd_maxq_heatmap_s{ngd_params["s"]}.png', dpi=600)
    plt.show()

    

def compare_sim(params, q2_state, analytic=False):
    """
    Given a set of parameters
    Draw trajectory curves for gene drive and non-gene drive simulations.
    """
    # Run the gene drive simulation
    gd = False
    params['q1'] = 0.1
    params['q2'] = 0.0
    gd_params = params.copy()
    ngd_params = params.copy()
    gd_params['alpha'] = 1.12
    gd_params['beta'] = 0.75
    gd_params['m'] = ngd_params['m'] = 0.01
    genotype = params['type']
    # set s1 s2 in gene drive params and find se1 (he1)
    s1, s2 = gd_params['s'], round(gd_params['s']*gd_params['alpha'], 2)
    c2 = round(gd_params['c']*gd_params['beta'],2)
    print("s1, s2:", s1, s2)
    c, h = gd_params['c'], gd_params['h']
    if q2_state == 'unstable':
        q_eq = get_unstable_qeq(gd_params)
        print("Unstable q_eq:", q_eq)
    gd_result = gd_simulation(gd_params)

    tl = 400

    for key in ['q1', 'q2']:
        if len(gd_result[key]) < tl:
            gd_result[key] = np.pad(gd_result[key], (0, tl - len(gd_result[key])), 'edge')
        elif len(gd_result[key]) > tl:
            gd_result[key] = gd_result[key][:tl]
    ### PLOT GD ONLY
    if gd: 
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(gd_result['q1'], label=f'Gene Drive Allele (q1) (m={gd_params['m']}, alpha={gd_params['alpha']})')
        plt.plot(gd_result['q2'], label='Wild-type Allele (q2)')
        plt.title(f'Gene Drive Simulation (s1={gd_params["s"]}, s2={gd_params["s"]*gd_params['alpha']:.3f}, c={gd_params["c"]}, h={gd_params["h"]})')
        plt.xlabel('Generations')
        plt.ylabel('Allele Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'gd_unstable_traj_fig/q{params['q1']}_{params['type']}_s{gd_params['s']}_c{gd_params['c']}_h{gd_params['h']}_m{gd_params['m']}_alpha{gd_params['alpha']}.png', dpi=600)
        plt.show()
        return 
    mapped_config = find_mapped(s1, c, h, genotype, analytic)
    if mapped_config == None:
        print(f"Error in s1 fixation mapping: no mapping found for s1={s1}")
        if not analytic:
            mapped_config = read_loss_unstable_res(s1, c, h, q2_state) # diploid loss mapping res
        else:
            mapped_config = find_unstable_mapped_analytic(s1, c, h)
    if genotype == "diploid":
        se1, he1 = mapped_config
        ngd_params['h'] = he1
    else:
        se1 = mapped_config
    ngd_params['s'] = se1
    print("deme 1 mapped config:", se1, he1)
    
    ### PLOT NGD FROM MAPPING 
    # ngd_params['m'], ngd_params['alpha'] = find_m_alpha(ngd_params)

    # find se2 (he2) from loss mapping
    ngd_d2_config = read_loss_unstable_res(s2, c2, h, q2_state)
    # ngd_d2_config = (np.float64(0.89), np.float64(0.63))
    se2, he2 = ngd_d2_config

    ngd_params['alpha'] = se2/se1
    ngd_params['type'] = "diploid"
    ngd_params['h'] = (he1, he2)
    print(gd_params, ngd_params)

    ngd_result = ngd_simulation(ngd_params)
    for key in ['q1', 'q2']:
        if len(ngd_result[key]) < tl:
            ngd_result[key] = np.pad(ngd_result[key], (0, tl - len(ngd_result[key])), 'edge')
        elif len(ngd_result[key]) > tl:
            ngd_result[key] = ngd_result[key][:tl]

    # Plot the results
    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(gd_result['q1'][:200], label=f'Gene Drive Allele (q1) (m={gd_params['m']}, alpha={gd_params['alpha']:.3f}, beta={gd_params['beta']:.3f})')
    # plt.plot(gd_result['q2'][:200], label='Wild-type Allele (q2)')
    # plt.xlabel('Generations')
    # plt.ylabel('Allele Frequency')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(ngd_result['q1'][:200], label=f'Favoured Allele (q1) (m={ngd_params['m']}, alpha={ngd_params['alpha']:.3f},beta={ngd_params['beta']:.3f})')
    # plt.plot(ngd_result['q2'][:200], label='Other Allele (q2)')
    # plt.xlabel('Generations')
    # plt.ylabel('Allele Frequency')
    
    plt.subplot(1, 2, 1)
    plt.plot(gd_result['q1'], label=f'Gene Drive Allele D1 (q1) (m={gd_params['m']}, alpha={gd_params['alpha']:.3f}, beta={gd_params['beta']:.3f})')
    plt.plot(gd_result['q2'], label='Gene Drive Allele D2 (q2)')
    plt.title(f'Gene Drive Simulation (s1={gd_params["s"]:.2f}, s2={gd_params["s"]*gd_params['alpha']:.3f}, c={gd_params["c"]}, h={gd_params["h"]})')
    plt.xlabel('Generations')
    plt.ylabel('Allele Frequency')
    plt.legend(loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(ngd_result['q1'], label=f'Mutant Allele D1(q1) (m={ngd_params['m']}, alpha={ngd_params['alpha']:.3f},beta={ngd_params['beta']:.3f})')
    plt.plot(ngd_result['q2'], label='Mutant Allele D2(q2)')
    paramstr = f"s1={ngd_params["s"]:.2f}, s2={(ngd_params["s"]*ngd_params['alpha']):.2f}, h1={he1:.2f}, h2={he2:.2f}" if params['type'] == "diploid" else f"s1={ngd_params["s"]}, s2={(ngd_params["s"]*ngd_params['alpha']):.3f}"
    plt.title(f'NGD {params['type']} simulation ({paramstr})')
    plt.xlabel('Generations')
    plt.ylabel('Allele Frequency')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'/Users/wanbo/Desktop/paper_figures/q{params['q1']}_{params['type']}_gd_vs_ngd_s{gd_params['s']}_c{gd_params['c']}_h{gd_params['h']}_m{gd_params['m']}_alpha{gd_params['alpha']}.pdf', dpi=600)
    plt.show()

def explore_values(params):
    '''
    Run ngd simulation for the mapped parameters to explore the (m, alpha) space
    Search for the values of m and alpha that lead to gene swamping
    '''
    gd_params = params.copy()
    ngd_params = params.copy()
    mapped_config = find_mapped(gd_params['s'], gd_params['c'], gd_params['h'])
    ngd_params['s'] = mapped_config[0] 
    if params['type'] == "diploid":
        ngd_params['h'] = mapped_config[1]
    m_range, alpha_range = (0.01, 1.0), (0.01, 2.0) 
    resolution = 100
    m_vals = np.linspace(m_range[0], m_range[1], resolution)
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], resolution)
    gs_mesh = np.zeros((len(m_vals), len(alpha_vals)))
    for i, m_val in enumerate(m_vals):
        for j, alpha_val in enumerate(alpha_vals):
            ngd_params['m'] = m_val
            ngd_params['alpha'] = alpha_val
            ngd_result = ngd_simulation(ngd_params)
            if math.isclose(ngd_result['q1'][-1], 0, rel_tol=1e-6) and math.isclose(ngd_result['q2'][-1], 0, rel_tol=1e-6):
                gs_mesh[i, j] = 1
    # gs_mesh[i, j] == 1 if gene swamping occurs at (m_vals[i], alpha_vals[j]), else 0
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("Blues")
    plt.imshow(gs_mesh, extent=(alpha_range[0], alpha_range[1], m_range[0], m_range[1]),
               aspect='auto', origin='lower', cmap=cmap)
    legend_elements = [
        Patch(facecolor=cmap(0), edgecolor='black', label='No Gene Swamping'),
        Patch(facecolor=cmap(1.0), edgecolor='black', label='Gene Swamping')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Gene Swamping Partition")
    plt.xlabel('Alpha (Ratio of Relative Fitness)')
    plt.ylabel('Migration Rate (m)')
    plt.title('Gene Swamping Occurrence in Non-gene Drive Simulation')
    plt.show()

# def find_alpha_range(s, c, h, state = "loss"):
#     # s1_max = float(s_q3_0_lambda(c, h))
#     if state == "loss": 
#         s2_bound = float(s_q3_1_lambda(c, h))
#     elif state == "unstable": 
#         s2_bound = float(unstable_lambda(c, h))
#     alpha_low = s2_bound/s
#     alpha_hi = 1.0/s # is alpha positive or negative? 
#     lo, hi = alpha_low,alpha_hi
#     print(f"lo:{lo}, hi:{hi}")


#     if lo >= hi:
#         raise ValueError(f"No valid s1: lower={lo:.4f} ≥ upper={hi:.4f}")
#     return lo, hi

def loss_mapping(s, c, h):
    '''
    Find mapping for a gene drive configuration in the loss regime
    q_init = 0.001
    All ngd_results are read from previous saved files
    Each gd_curve is found through running the gd simulation

    08.11 Update: We do not use this anymore with OSG mapping results
    '''
    gd_results = load_pickle(f"gd_simulation_results/h{h}_allgdres001G.pickle")
    # stabilityRes = load_pickle(f"../pickle/h{h}_gametic_stability_res.pickle")
    # ngd_results = load_pickle(f"allngdres001G.pickle")
    gd_configs, gd_res = gd_results[0], gd_results[1]
    # gd_curve = gd_res[(s, c, h)]['q']
    params = {'s': s, 'c': c, 'h': h, 'target_steps': 40000, 'q0': 0.001}
    gd_curve = run_model(params)['q']
    best_ngd_config = dict()
    best_ngd_curve = None
    q_init_list = [0.001, 0.2, 0.5, 0.8]
    for q_init in q_init_list:
        best_ngd_config_q = None
        best_diff = 10000
        params['q0'] = q_init
        for ngd_s in np.arange(0.01, 1.01, 0.01):
            ngd_params = {'s': ngd_s, 'target_steps': 40000, 'q0': q_init}
            ngd_curve = haploid(ngd_params)['q']
            gd_curve = run_model(params)['q']
            diff = euclidean(ngd_curve, gd_curve)
            if diff < best_diff:
                best_diff = diff
                best_ngd_config_q = ngd_s
                # best_ngd_curve_q = ngd_curve
        best_ngd_config[q_init] = (best_ngd_config_q, best_diff)
        # best_ngd_curve[q_init] = best_ngd_curve_q
    with open(f"s{s}_c{c}_h{h}_loss_output.txt", "w") as f:
        for q_init, (ngd_s, best_diff) in best_ngd_config.items():
            f.write(f"q = {q_init}: {ngd_s}, diff = {best_diff}\n")
    plot_loss_mapping(params, best_ngd_config, q_init_list)
    return best_ngd_config

def plot_loss_mapping(params, best_ngd_config, q_init_list):
    plt.figure(figsize=(10, 6))
    for q_init in q_init_list:
        params['q0'] = q_init
        ngd_s, diff = best_ngd_config[q_init]
        gd_curve = run_model(params)['q']
        ngd_params = {'s': ngd_s, 'target_steps': 40000, 'q0': q_init}
        ngd_curve = haploid(ngd_params)['q']
        plt.plot(gd_curve, label=f'Gene Drive Curve(q0={q_init:.3f})', color='blue')
        plt.plot(ngd_curve, label=f'NGD Curve (q0={q_init:.3f}, s={ngd_s:.3f}, MSE={diff})', linestyle='--')
    
    plt.title('Loss Mapping: Gene Drive vs Non-Gene Drive Curves')
    plt.xlabel('Generations')
    plt.ylabel('Allele Frequency')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def gd_swamping_table(params,
                    m_vals=None,
                    alpha_vals=None,
                    n_m=50,
                    n_alpha=50,
                    tol=1e-2):
    # unpack constants
    s1 = params['s'] # fixation regime
    c = params['c']
    h = params['h']
    
    if m_vals is None:
        m_vals = np.linspace(0.01, 1.0, n_m)
    if alpha_vals is None:
        lo, hi = find_alpha_range(s1, c, h, "unstable")
        alpha_vals = np.linspace(lo, hi, n_alpha)
    rows = []
    for m in m_vals:
        for alpha in alpha_vals:
            s2 = s1 * float(alpha)  # s2 in unstable regime by construction

            # Build per-run params, set s2 via alpha, and set q_init = q_eq^unstable
            gd_p = params.copy()
            gd_p.update({
                'm':     float(np.round(m, 6)),
                'alpha': float(np.round(alpha, 6)),
            })

            # Get unstable equilibrium for THESE parameters
            q_eq = get_unstable_qeq(gd_p)

            # Initialize BOTH demes at the unstable equilibrium
            gd_p['q1'] = 0.1
            gd_p['q2'] = 0.01

            # Run GD simulation
            gd_out = gd_simulation(gd_p)

            # Gene swamping if both q1 and q2 go to ~0 (use your project’s checker)
            gs_gd = check_gs_traj(gd_out)

            rows.append({
                'm':                float(np.round(m, 4)),
                's1':               float(np.round(s1, 4)),
                's2':               float(np.round(s2, 4)),
                'c':                float(np.round(c, 4)),
                'h':                float(np.round(h, 4)) if h is not None else None,
                'alpha':            float(np.round(alpha, 4)),
                'q_eq_unstable':    float(np.round(q_eq, 6)),
                'gene_swamping_gd': int(gs_gd),
            })

    return pd.DataFrame(rows)

def plot_gs_alphae(plot_df, type='ngd'):
    # Filter by fixed s1 value (selection in deme 1)
    # plot_df = df[df['s1'].round(4) == round(s_fixed, 4)]
    s_fixed = plot_df['s1'].iloc[0]
    c = plot_df['c'].iloc[0]
    h = plot_df['h'].iloc[0]

    # Create pivot table for pcolormesh (must be 2D grid)
    pivot = plot_df.pivot(index='m', columns='alpha', values='alpha_e')
    m_vals = pivot.index.values
    alpha_vals = pivot.columns.values
    Z = pivot.values

    # Create meshgrid for pcolormesh
    alpha_grid, m_grid = np.meshgrid(alpha_vals, m_vals)

    fig, ax = plt.subplots(figsize=(8, 6))

    # pcolormesh expects the grid edges, so we need to extend the grid by half-step
    # Calculate step sizes
    if len(alpha_vals) > 1:
        dalpha = (alpha_vals[1] - alpha_vals[0]) / 2
    else:
        dalpha = 0.05
    if len(m_vals) > 1:
        dm = (m_vals[1] - m_vals[0]) / 2
    else:
        dm = 0.05

    alpha_edges = np.concatenate(([alpha_vals[0] - dalpha], (alpha_vals[:-1] + alpha_vals[1:]) / 2, [alpha_vals[-1] + dalpha]))
    m_edges = np.concatenate(([m_vals[0] - dm], (m_vals[:-1] + m_vals[1:]) / 2, [m_vals[-1] + dm]))

    cmap = plt.get_cmap("viridis")
    # Plot using pcolormesh
    mesh = ax.pcolormesh(
        alpha_edges,
        m_edges,
        Z,
        cmap=cmap,
        edgecolors='k',
        linewidth=0.2,
        shading='auto'
    )
    cbar = plt.colorbar(mesh, ax=ax, label='alpha_e (effective alpha)')

    # Style the plot
    title = "(NGD) mapped from" if type == "ngd" else "(GD) with"
    ax.set_title(f"Alpha_e {title} S_gd={s_fixed}, C_gd={c}, H_gd={h}")
    ax.set_xlabel("α (alpha in GD)")
    ax.set_ylabel("m (migration rate)")
    ax.set_xlim(plot_df['alpha'].min(), plot_df['alpha'].max())
    ax.set_ylim(plot_df['m'].min(), plot_df['m'].max())

    # Add custom legend
    # from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor=cmap(1.0), edgecolor='k', label='Swamping (1)'),
    #     Patch(facecolor=cmap(0), edgecolor='k', label='No Swamping (0)')
    # ]
    # ax.legend(handles=legend_elements, title="Gene Drive")
    plt.show()
    plt.grid(False)
    fig.savefig(f"gs_alphae_s{s_fixed}_c{c}_h{h}_{type}_diploid.png", dpi=600)

    return ax

# 08.03 Updates:
# Plot figures showing the partition for a gene drive and a mapped ngd model 
# Generate table with varying m and alpha, and each row should show whether there is gs in either model
# Loss mapping should now read from the acquired result in all_loss_mapping_res.pickle

# 09.22 Updates:
# added d2 unstable state for read_loss_unstable_res
# added analytic implementations
# added case 3 (restricted alpha) for d2 unstable/loss
# TO-DO: finish case 2 (changing c instead of s from deme 1 to deme 2)
# 
# 
# START OF GENE SWAMPING TABLE AND FIGURES FUNCTIONS #################################
def _eval_candidates(funcs, cval, hval):
    vals = []
    for fn in funcs:
        try:
            v = float(fn(cval, hval))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return vals

def _internal_qstar_exists_and_unstable(s2, cval, hval):
    try:
        qstar = float(q3_func(s2, cval, hval))
        # print("Checking qstar in range:", qstar, "for s2:", s2, "c:", cval, "h:", hval)
        if not (0.0 < qstar < 1.0) or not np.isfinite(qstar):
            return False
        der = float(df_at_q3_func(s2, cval, hval))
        return np.isfinite(der) and abs(der) > 1.0
    except Exception:
        return False

def find_beta_range(s, c, h, eps=1e-6):
    beta_min = eps
    beta_max = 1.0
    step = int((beta_max-beta_min)/0.01) 
    beta_grid = np.linspace(beta_min, beta_max, step) 
    
    c2_cands0 = _eval_candidates(c_q3_0_funcs, s, h)
    c2_cands1 = _eval_candidates(c_q3_1_funcs, s, h)
    if not c2_cands0 or not c2_cands1:
        raise ValueError(f"Cannot bracket internal root with q3=0/1 collisions for these ({s},{h}).")
    c2_min = max(min(c2_cands0 + c2_cands1), 0.0)
    c2_max = min(max(c2_cands0 + c2_cands1), 1.0)

    # 2) Translate to alpha via s2 = -alpha*s1
    # We’ll scan alpha and test instability numerically (robust to weird symbolic branches)
    mask = []
    for a in beta_grid:
        c2 = a * c
        # print("Testing s2:", s2, "from alpha:", a)
        in_span = (c2_min <= c2 <= c2_max) or (c2_max <= c2 <= c2_min)
        mask.append(in_span and _internal_qstar_exists_and_unstable(s, c2, h))
    mask = np.array(mask, dtype=bool)
    # print(mask)

    if not mask.any():
        raise ValueError("No beta range produces an internal unstable fixed point at these (s1,c,h).")

    # 3) Return the first contiguous interval where condition holds
    idx = np.where(mask)[0]
    lo = float(beta_grid[idx[0]])
    hi = float(beta_grid[idx[-1]])

    return lo, hi


def find_alpha_range(s1, cval, hval, case="s1", state="loss", alpha_grid=None):
    # sensible alpha scan if none provided
    if alpha_grid is None:
        alpha_min = 0.01
        alpha_max = min(1.2, 1.0/s1) if case=="s3" else 1.0/s1
        step = int((alpha_max-alpha_min)/0.01)
        alpha_grid = np.linspace(alpha_min, alpha_max, step)   # adjust if you need wider

    if state == "loss":
        # your previous approach: use the q3=1 collision as a bound
        s2_bounds = _eval_candidates(s_q3_1_funcs, cval, hval)
        print(s2_bounds)
        if not s2_bounds:
            raise ValueError("No q3=1 bound found for these (c,h).")
        s2_bound = s2_bounds[0]  
        alpha_lo = s2_bound / s1    # note: s2 negative if alpha>0, so signs matter
        alpha_hi = min(1.2, 1.0 / s1) if case == "s3" else 1.0/s1       # if this is part of your existing convention
        lo, hi = alpha_lo, alpha_hi

    elif state == "unstable":
        # 1) Find the s2-span where an internal root can exist (between q3=0 and q3=1 collisions)
        s2_cands0 = _eval_candidates(s_q3_0_funcs, cval, hval)
        s2_cands1 = _eval_candidates(s_q3_1_funcs, cval, hval)
        if not s2_cands0 or not s2_cands1:
            raise ValueError("Cannot bracket internal root with q3=0/1 collisions for these (c,h).")
        s2_min = max(min(s2_cands0 + s2_cands1), 0.0)
        s2_max = min(max(s2_cands0 + s2_cands1), 1.0)

        # 2) Translate to alpha via s2 = -alpha*s1
        # We’ll scan alpha and test instability numerically (robust to weird symbolic branches)
        mask = []
        for a in alpha_grid:
            s2 = a * s1
            # print("Testing s2:", s2, "from alpha:", a)
            in_span = (s2_min <= s2 <= s2_max) or (s2_max <= s2 <= s2_min)
            mask.append(in_span and _internal_qstar_exists_and_unstable(s2, cval, hval))
        mask = np.array(mask, dtype=bool)
        # print(mask)

        if not mask.any():
            raise ValueError("No alpha range produces an internal unstable fixed point at these (s1,c,h).")

        # 3) Return the first contiguous interval where condition holds
        idx = np.where(mask)[0]
        lo = float(alpha_grid[idx[0]])
        hi = float(alpha_grid[idx[-1]])

    else:
        raise ValueError("state must be 'loss' or 'unstable'")

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"No valid alpha: lower={lo} ≥ upper={hi}")
    print(lo, hi)
    return lo, hi


def read_loss_unstable_res(s, c, h, state):
    '''
    Open the OSG loss/unstable mapping result file and load the mapping result for the input config
    returns s_e, h_e
    '''
    # First run the combine_loss_res.py script to generate the all_loss_mapping_res_diploid.pickle file
    if state == "loss":
        with open("../OSG_submit/all_loss_mapping_res_diploid.pickle", "rb") as pf:
            result = pickle.load(pf)
        if (s, c, h) not in result:
            print(f"Error: GD config ({s}, {c}, {h}) not in loss regime")
            return None
        s_ngd, h_ngd, mse = result[(s, c, h)]
    elif state == "unstable":
        with open(f"../scripts/unstable_mapping_result/unstable_combined_osg/pickle/h{h}_unstable_grid_all.pickle", "rb") as pf:
            result = pickle.load(pf)
        if (s, c, h) not in result:
            print(f"Error: GD config ({s}, {c}, {h}) not in unstable regime")
            return None
        s_ngd, h_ngd, mse = result[(s, c, h)]
    return (s_ngd, h_ngd)

def is_swamping(q, tol=1e-3, decay_thresh=0.1):
    """Check if a single deme is swamping toward 0."""
    if len(q) <= 1 and not abs(q[-1]) < tol:
        print(f"Warning: Trajectory {q} too short to determine swamping.")
        return False
    # Already essentially zero
    if abs(q[-1]) < tol:
        return True
    # Still decreasing and below a small threshold
    if (q[-1] < q[-2] and not math.isclose(q[-1], q[-2]) 
            and q[-1] < decay_thresh):
        return True
    return False

def check_gs_traj(traj, tol=1e-3, decay_thresh=0.1):
    """Check gene swamping: both demes trending to 0 or below tolerance."""
    return is_swamping(traj['q1'], tol, decay_thresh) and \
           is_swamping(traj['q2'], tol, decay_thresh)
    

def get_unstable_qeq(p):
        """Resolve q* if qeq_func not provided."""
        s, c, h, alpha = p['s'], p['c'], p['h'], p['alpha']
        s2 = s*alpha
        if q3_func is not None:
            return float(q3_func(s2, c, h))
    
        raise ValueError("No unstable equilibrium resolver found. ")

def find_unstable_mapped_analytic(s, c, h):
    denom = 2*c*h*s-2*c+s
    if math.isclose(denom, 0.0):
        print("zero denom in analytic")
        return None
    h_ngd = (c*h*s-c+h*s)/(2*c*h*s-2*c+s)
    s_ngd = 2*c*h*s-2*c+s
    return (s_ngd, h_ngd)

def generate_swamping_table(params,
                            d2_state,
                            analytic,
                            case="s1",
                            s2=False,
                            n_m=50,
                            n_alpha=50,   # kept name for backward compat; used for beta too
                            tol=1e-2):
    """
    If params['vary_mode'] == 'alpha' (default):
        fix (s1, c, h); sweep alpha; s2 = s1*alpha
    If params['vary_mode'] == 'beta':
        fix (s, c1, h); sweep beta; c2 = c1*beta
    """
    vary_mode = params.get('vary_mode', 'alpha').lower()
    if vary_mode not in ('alpha', 'beta'):
        raise ValueError("params['vary_mode'] must be 'alpha' or 'beta'.")

    # Unpack constants
    s1 = params['s']
    c1 = params['c']
    h  = params.get('h', None)

    # m grid (unchanged)
    m_vals = np.linspace(0.01, 1.0, n_m)

    # Build the sweep grid depending on mode
    if vary_mode == 'alpha':
        # old behavior
        lo, hi = find_alpha_range(s1, c1, h, case, d2_state)
        n_alpha = max(2, int(abs((hi - lo) / 0.01)))
        sweep_vals = np.linspace(lo, hi, n_alpha)
        sweep_name = 'alpha'
    else:
        # beta sweep: pick a reasonable range; feel free to tighten later
        # e.g., let c2 in (0,1]; if c1 in (0,1], then beta in (eps, 1/c1]
        eps = 1e-3
        lo, hi = find_beta_range(s1, c1, h, eps)  # cap to avoid crazy big c2
        sweep_vals = np.linspace(lo, hi, n_alpha)
        sweep_name = 'beta'

    rows = []

    # Map the fixed deme 1 GD config once (this is (s1, c1, h) in both modes)
    s1_mapped = find_mapped(s1, c1, h, params['type'], analytic)
    if s1_mapped is None:
        print(f"Error in s1 fixation mapping: no mapping found for s1={s1}")
        if not analytic:
            s1_mapped = read_loss_unstable_res(s1, c1, h, d2_state)
        else:
            s1_mapped = find_unstable_mapped_analytic(s1, c1, h)
    if s1_mapped is None:
        raise RuntimeError("Failed to map deme 1; cannot proceed.")
    se1, he1 = s1_mapped

    # Precompute the mapped (se2, he2) per sweep value
    mapped_d2 = {}  # key: sweep value -> (se2, he2, alpha_e)
    for v in sweep_vals:
        if vary_mode == 'alpha':
            alpha = float(v)
            s2 = round(float(s1 * alpha), 2)
            c2 = round(c1, 2)
        else:
            beta = float(v)
            s2 = round(float(s1), 2)
            c2 = round(float(c1 * beta), 2)

        # Map (s2, c2, h) via your mapping
        if not analytic:
            mapped = read_loss_unstable_res(s2, c2, h, d2_state)
        else:
            mapped = find_unstable_mapped_analytic(s2, c2, h)
        if mapped is None:
            continue
        se2, he2 = mapped

        # Effective alpha_e stays defined as se2/se1
        alpha_e = se2 / se1 if se1 != 0 else np.nan

        key = alpha if vary_mode == 'alpha' else beta
        mapped_d2[key] = (se2, he2, alpha_e)

    # Main sweep over (m, sweep_val)
    for m_val in m_vals:
        for v, (se2, he2, alpha_e) in mapped_d2.items():
            if vary_mode == 'alpha':
                alpha = float(v)
                s2    = float(s1 * alpha)
                c2    = c1
            else:
                beta  = float(v)
                s2    = float(s1)
                c2    = float(c1 * beta)

            # --- GD simulation ---
            gd_p = params.copy()
            gd_p.update({
                'm':     round(m_val, 3),
                'alpha': round(alpha if vary_mode=='alpha' else (s2/s1 if s1!=0 else np.nan), 6),  # keep 'alpha' column for plotting if desired
                'c':     c1,      # GD deme-1 c is c1
            })
            # Deme-2 GD parameter is either s2 (alpha mode) or c2 (beta mode).
            # If your gd_simulation reads s via 's' + 'alpha', you’re set.
            # If it needs c2 explicitly, include it in gd_p as well:
            gd_p['beta'] = c2/c1   # <- only used if your simulator supports per-deme c

            gd_out = gd_simulation(gd_p)
            gs_gd = check_gs_traj(gd_out)

            # --- NGD simulation ---
            ngd_p = params.copy()
            ngd_p.update({
                's':     se1,
                'h':     (he1, he2),   # per-deme dominance if your NGD code supports tuple
                'm':     m_val,
                'alpha': alpha_e,      # still ratio of mapped s's
                'type':  "diploid"
            })
            ngd_out = ngd_simulation(ngd_p)
            gs_ngd = check_gs_traj(ngd_out)

            # Rounding & bookkeeping
            s1_r = round(s1, 6)
            s2_r = round(s2, 6)
            c1_r = round(c1, 6)
            c2_r = round(c2, 6)
            # alpha_r = round(alpha, 4)
            # beta_r = round(beta, 4)
            val_r = round(float(v), 6)
            se1_r = round(se1, 6)
            se2_r = round(se2, 6)
            he1_r = round(he1, 6) if he1 is not None else None
            he2_r = round(he2, 6) if he2 is not None else None
            alpha_e_r = round(alpha_e, 6) if alpha_e is not None else None

            row = {
                'm':                  round(m_val, 6),
                's1':                 s1_r,
                's2':                 s2_r,
                'c1':                 c1_r,
                'c2':                 c2_r,
                'c':                  c1_r,   # preserve old column 'c' for downstream code
                'h':                  round(h, 6) if h is not None else None,
                # 'alpha':              alpha_r,
                # 'beta':               beta_r,
                'gene_swamping_gd':   gs_gd,
                'se1':                se1_r,
                'se2':                se2_r,
                'he1':                he1_r,
                'he2':                he2_r,
                'alpha_e':            alpha_e_r,
                'gene_swamping_ngd':  gs_ngd,
                'vary_mode':          vary_mode,
            }
            # Keep both columns so plotting can choose
            if vary_mode == 'alpha':
                row['alpha'] = val_r
                row['beta']  = np.nan
            else:
                row['beta']  = val_r
                # also store the implied alpha = s2/s1 to avoid breaking old plots
                row['alpha'] = round((s2/s1) if s1 != 0 else np.nan, 6)

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def _edges_from_centers(vals):
    vals = np.asarray(vals)
    if vals.size == 1:
        step = 0.05
        return np.array([vals[0]-step, vals[0]+step])
    steps = np.diff(vals) / 2.0
    return np.concatenate(([vals[0]-steps[0]], vals[:-1]+steps, [vals[-1]+steps[-1]]))

def _pivot_on_grid(df, value_col, m_vals, alpha_vals):
    # Pivot then reindex to common axes so both GD/NGD align
    pv = df.pivot(index='m', columns='ratio', values=value_col)
    pv = pv.reindex(index=m_vals, columns=alpha_vals)
    return pv.values

def plot_gs_partition_overlap(df, q0=None, state="unstable", analytic=False, case="s1"):
    vary_mode = df['vary_mode'].iloc[0] if 'vary_mode' in df.columns else 'alpha'
    xcol = 'alpha' if vary_mode == 'alpha' else 'beta'

    gd_df  = df[['m', xcol, 'gene_swamping_gd']].dropna()
    ngd_df = df[['m', xcol, 'gene_swamping_ngd']].dropna()

    s_fixed = float(df['s1'].iloc[0])
    c = float(df['c'].iloc[0]) if 'c' in df.columns else float(df['c1'].iloc[0])
    h = float(df['h'].iloc[0])

    # Build grids
    m_vals = np.sort(gd_df['m'].unique())
    x_vals = np.sort(gd_df[xcol].unique())

    # Align & plot (same as your code, but using xcol/x_vals)
    Z_gd  = _pivot_on_grid(gd_df.rename(columns={xcol: 'ratio'}),  'gene_swamping_gd',  m_vals, x_vals)
    Z_ngd = _pivot_on_grid(ngd_df.rename(columns={xcol: 'ratio'}), 'gene_swamping_ngd', m_vals, x_vals)

    m_edges = _edges_from_centers(m_vals)
    a_edges = _edges_from_centers(x_vals)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    colors = ["#32B7DC", "#e84ade"]
    custom_cmap_ngd = ListedColormap(["#ffffff", colors[0]])
    custom_cmap_gd  = ListedColormap(["#ffffff", colors[1]])

    ax.pcolormesh(a_edges, m_edges, Z_ngd, cmap=custom_cmap_ngd, vmin=0, vmax=1, shading="auto", alpha=0.6, linewidth=0.0)
    ax.pcolormesh(a_edges, m_edges, Z_gd,  cmap=custom_cmap_gd,  vmin=0, vmax=1, shading="auto", alpha=0.2, linewidth=0.0)

    a = "Analytic" if analytic else 'Simulation'
    xlabel = "α (alpha)" if vary_mode == 'alpha' else "β (c₂ / c₁)"
    ax.set_title(f"Gene Swamping Partition (GD vs NGD {a})\nGD: s={s_fixed}, c={c}, h={h}")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("m (migration rate)", fontsize=14)
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(m_vals.min(), m_vals.max())
    ax.grid(False)

    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='k', label='NGD: swamping'),
        Patch(facecolor=colors[1], edgecolor='k', label='GD: swamping')]
    ax.legend(handles=legend_elements, title="Outcome", loc='best', frameon=True)

    fig.tight_layout()
    ana = "analytic" if analytic else "sim"
    foldername = f"{case}_sim_partition_fig"
    fig.savefig(f"{foldername}/{state}_{ana}_overlap_s{s_fixed}_c{c}_h{h}_{vary_mode}.pdf", dpi=600)
    plt.show()
    return ax


### END GS PARTITION OVERLAP FIGURE ###################################


def plot_gs_partition(plot_df, q0, type, state, analytic):
    ana = "analytic" if analytic else "sim"
    # Filter by fixed s1 value (selection in deme 1)
    # plot_df = df[df['s1'].round(4) == round(s_fixed, 4)]
    s_fixed = plot_df['s1'].iloc[0]
    c = plot_df['c'].iloc[0]
    h = plot_df['h'].iloc[0]

    # Create pivot table for pcolormesh (must be 2D grid)
    pivot = plot_df.pivot(index='m', columns='alpha', values=f'gene_swamping_{type}')
    m_vals = pivot.index.values
    alpha_vals = pivot.columns.values
    Z = pivot.values

    # Create meshgrid for pcolormesh
    alpha_grid, m_grid = np.meshgrid(alpha_vals, m_vals)

    fig, ax = plt.subplots(figsize=(8, 6))

    # pcolormesh expects the grid edges, so we need to extend the grid by half-step
    # Calculate step sizes
    if len(alpha_vals) > 1:
        dalpha = (alpha_vals[1] - alpha_vals[0]) / 2
    else:
        dalpha = 0.05
    if len(m_vals) > 1:
        dm = (m_vals[1] - m_vals[0]) / 2
    else:
        dm = 0.05

    alpha_edges = np.concatenate(([alpha_vals[0] - dalpha], (alpha_vals[:-1] + alpha_vals[1:]) / 2, [alpha_vals[-1] + dalpha]))
    m_edges = np.concatenate(([m_vals[0] - dm], (m_vals[:-1] + m_vals[1:]) / 2, [m_vals[-1] + dm]))

    cmap = plt.get_cmap("Blues")
    # Plot using pcolormesh
    mesh = ax.pcolormesh(
        alpha_edges,
        m_edges,
        Z,
        cmap=cmap,     # Black = 1, White = 0
        edgecolors='k',
        linewidth=0.2,
        shading='auto'
    )

    # Style the plot
    title = "(NGD) mapped from" if type == "ngd" else "(GD) with"
    ax.set_title(f"Gene Swamping Partition {title} S_gd={s_fixed}, C_gd={c}, H_gd={h}")
    ax.set_xlabel("α (alpha)")
    ax.set_ylabel("m (migration rate)")
    ax.set_xlim(plot_df['alpha'].min(), plot_df['alpha'].max())
    ax.set_ylim(plot_df['m'].min(), plot_df['m'].max())

    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(1.0), edgecolor='k', label='Swamping (1)'),
        Patch(facecolor=cmap(0), edgecolor='k', label='No Swamping (0)')
    ]
    ax.legend(handles=legend_elements, title="Gene Drive")
    plt.show()
    plt.grid(False)
    fig.savefig(f"gd_sim_partition_fig/{state}_{ana}_s{s_fixed}_c{c}_h{h}_{type}_partition.png", dpi=600)

    return ax

def plot_eq_alpha(params, m_vals, deme, n_alpha=100):
    '''
    Plot the final value of q_gd in deme 1/2 at different alpha at the end of simulation.
    Each curve corresponds to a fixed migration rate (m).
    '''
    s1 = params['s']  # fixation regime
    c = params['c']
    h = params.get('h', None)
    deme = str(deme)  # Ensure deme is a string

    lo, hi = find_alpha_range(s1, c, h)
    alpha_vals = np.linspace(lo, hi, n_alpha)

    plt.figure(figsize=(8, 6))

    for m in m_vals:
        final_qs = []
        for alpha in alpha_vals:
            params["alpha"] = alpha
            params["m"] = m
            s2 = s1 * alpha  # s2 in gd loss regime
            s2 = round(float(s2), 2)
            params.update({
                'alpha': round(alpha, 4),
            })
            gd_out = gd_simulation(params)
            final_qs.append(gd_out[f'q{deme}'][-1])  # Get the final q value for the specified deme

        plt.plot(alpha_vals, final_qs, label=f'm={m:.3f}')

    plt.title(f'Final q in gene drive model deme {deme} vs alpha for different m values')
    plt.xlabel('Alpha (α)')
    plt.ylabel(f'Final q in deme {deme}')
    plt.legend(title=f's={s1},c={c},h={h}\n Migration Rate (m)', loc='lower right')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'gd_plots/q{params['q1']}_final_q{deme}_vs_alpha_s{s1}_c{c}_h{h}.png', dpi=600)
    plt.show()

def main():
    params = {
        'q1': 0.1,          # Initial frequency in population 1
        'q2': 0.0,          # Initial frequency in population 2
        'target_steps': 1000,
        's': 0.85,           # Selection coefficient in deme 1 (fixation)
        'c': 0.8,           # Conversion rate
        'h': 0.5,           # Dominance coefficient
        'm': 0.5,          # Migration rate
        'alpha': 5.0,        # Relative fitness of the favoured allele
        'beta': 1.0,
        'type': "diploid", 
        'vary_mode':'alpha'
    }
    # find_alpha_range(0.2, 0.3, 1.0, "unstable")
    # run_ngd_all(params) 
    # plot_ngd_single(params)  
    # print(loss_mapping(0.8, 0.2, 0.8)) 
    traj = True
    analytic = True
    case = "s3"
    d2_state = "loss"
    if traj: 
        compare_sim(params, d2_state, analytic)
    else: 
        # explore_values(params)
        s, c, h, m = params['s'], params['c'], params['h'], params['m'] 
        ana = "analytic" if analytic else "sim"
        folder = f"{case}_swamping_table"
        txtname = f"{folder}/gd_{d2_state}_{ana}_s{s}_c{c}_h{h}.txt"


        # Plot swamping parititon
        try:
            swamping_df = pd.read_csv(txtname, delim_whitespace=True)
            print("File loaded successfully:")
        except FileNotFoundError:
            print('ERROR: need to generate table first')
            table = generate_swamping_table(params, d2_state, analytic, case)
            # table = gd_swamping_table(params)
            print(table.head())
            # # # Save to CSV 
            table.to_string(txtname, index=False, justify='left')
            swamping_df = pd.read_csv(txtname, delim_whitespace=True)
        plot_gs_partition_overlap(swamping_df, params['q1'], d2_state, analytic, case)
        # plot_gs_partition(swamping_df, params['q1'], "gd", d2_state)
        # plot_gs_partition(swamping_df, params['q1'], "ngd", d2_state)

    # m_vals = np.linspace(0.01, 0.5, 10)
    # plot_eq_alpha(params, m_vals, "1")

# Thoughts? 
# Does the mapping depend on the accuracy of mapping??
# current gap: 
# What's currently unknown when looking at the gene drive model is that when given a 
# migration rate and the (s, c, h) setting, how heterogeneous should we make the fitness
# in 2 demes be such that there can be gene swamping that eliminate the gene drive allele
# ==> Does our approximations allow as to know something about this prediction if we can
# predict it with a more well-studied/simpler model? 


if __name__ == "__main__":
    main()
