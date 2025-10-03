# %%
import sympy
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers import load_pickle
from gd_model import gd_simulation, one_step

model = "ngd"
def get_analytics():
    #%%
    model = "ngd"
    type = "Diploid"
    q1 = sympy.Symbol('q1')
    q2 = sympy.Symbol('q2')
    s = sympy.Symbol('s')
    c = sympy.Symbol('c')
    h = sympy.Symbol('h')
    m = sympy.Symbol('m')
    alpha = sympy.Symbol('alpha')

    s2 = s * alpha ## selection coefficient in deme 2
    
    #%%
    # Migration
    q1_m = q1*(1-m) + q2*m
    q2_m = q2*(1-m) + q1*m
    # Selection
    if model == "ngd":
        if type == "Haploid":
            print("hap")
            wA1, wa1 = 1-s, 1
            wA2, wa2 = 1-s2, 1
            expr_d1 = q1_m*wA1/(q1_m*wA1+(1-q1_m)*wa1)
            expr_d2 = q2_m*wA2/(q2_m*wA2+(1-q2_m)*wa2)    
        if type == "Diploid":
        # Fitness parameters (a favored in deme 2, A favored in deme 1):
            wAA1, wAa1, waa1 = 1 - s,   1 - h*s,       1
            wAA2, wAa2, waa2 = 1 - s2,  1 - h*s2,      1
            # wAA1, wAa1, waa1 = sympy.Symbol('wAA1'), sympy.Symbol('wAa1'), sympy.Symbol('waa1')
            # wAA2, wAa2, waa2 = sympy.Symbol('wAA2'), sympy.Symbol('wAa2'), sympy.Symbol('waa2')


            # Selection
            expr_d1 = (q1_m**2 * wAA1 + q1_m * (1-q1_m) * wAa1) / (q1_m**2 * wAA1 + 2 * q1_m * (1-q1_m) * wAa1 + (1-q1_m)**2 * waa1)
            expr_d2 = (q2_m**2 * wAA2 + q2_m * (1-q2_m) * wAa2) / (q2_m**2 * wAA2 + 2 * q2_m * (1-q2_m) * wAa2 + (1-q2_m)**2 * waa2)
    # Gene Drive selection
    else: 
        # Gene Drive fitness: A is gene drive allele and is favored in deme 1
        wAA1, wAa1, waa1 = 1 - s,   1 - h*s,       1
        wAA2, wAa2, waa2 = 1 - s2,  1 - h*s2,      1
        # wAA1, wAa1, waa1 = sympy.Symbol('wAA1'), sympy.Symbol('wAa1'), sympy.Symbol('waa1')
        # wAA2, wAa2, waa2 = sympy.Symbol('wAA2'), sympy.Symbol('wAa2'), sympy.Symbol('waa2')

        # Selection
        s_n1, s_c1 = 0.5 * (1-c) * wAa1, c * wAA1
        s_n2, s_c2 = 0.5 * (1-c) * wAa2, c * wAA2
        expr_d1 = (q1_m**2 * wAA1 + 2 * q1_m * (1-q1_m) * (s_n1 + s_c1)) / (q1_m**2 * wAA1 + 2 * q1_m * (1-q1_m) * (2 * s_n1 + s_c1) + (1-q1_m)**2 * waa1)
        expr_d2 = (q2_m**2 * wAA2 + 2 * q2_m * (1-q2_m) * (s_n2 + s_c2)) / (q2_m**2 * wAA2 + 2 * q2_m * (1-q2_m) * (2 * s_n2 + s_c2) + (1-q2_m)**2 * waa2)

    # q_11, q_12, eq1 = sympy.solvers.solve(expr_d1 - q1, q1, dict=True)
    # q_21, q_22, eq2 = sympy.solvers.solve(expr_d2 - q2, q2, dict=True)
    eq1 = eq2 = 0

    #%%
    ################################################################################
    # 1) FULL SYMBOLIC JACOBIAN 
    ################################################################################
    J_sym = sympy.Matrix([[sympy.diff(expr_d1, q1),
                        sympy.diff(expr_d2, q1)],
                        [sympy.diff(expr_d1, q2),
                        sympy.diff(expr_d2, q2)]])

    ################################################################################
    # 2) SUBSTITUTE q1 = q2 = 0 
    ################################################################################
    subs_eq  = {q1: eq1, q2: eq2}
    J_eq_sym = sympy.simplify(J_sym.subs(subs_eq))

    print("\n--- Jacobian at the internal equilibrium (symbolic) ---")
    sympy.pretty_print(J_eq_sym)

    #%%
    ################################################################################
    # 3) SYMBOLIC EIGEN-VALUES  
    ################################################################################
    lam_eq_sym = sympy.Matrix(J_eq_sym).eigenvals()     
    print("\nEigen-values expressions:")
    for lam, mult in lam_eq_sym.items():
        print(f"λ ({mult}x) =", sympy.simplify(lam))

    #%%
    ################################################################################
    # 4) NUMERICAL DRIVER (lambdified) --------------------------------------------
    ################################################################################
    if model == "gd":
        free_syms = (s, h, m, c, alpha) 
    elif model == "ngd":
        if type == "Diploid":
            free_syms = (s, h, m, alpha)
        else: 
            free_syms = (s, m, alpha)

    # Lambdify Jacobian
    J_func = sympy.lambdify(free_syms, J_eq_sym, modules="numpy")

    # Lambdify eigen-values (as a column vector)
    leading_eig = list(lam_eq_sym.keys())[0]  
    print("Leading eigenvalue")
    sympy.pretty_print(leading_eig)
    # solve for when leading_eig = 1
    m_thresh_pos, m_thresh_neg = None, None
    if model == "ngd":
        pos_thresh = sympy.solvers.solve(leading_eig - 1, m, dict=True)
        neg_thresh = sympy.solve(leading_eig + 1, m, dict=True)
        m_thresh_pos = sympy.lambdify(free_syms, pos_thresh[0][m], 'numpy')
        m_thresh_neg = sympy.lambdify(free_syms, neg_thresh[0][m], 'numpy')
        print("critical m from solving leading eig = 1")
        sympy.pretty_print(sympy.simplify(pos_thresh[0]))
        m_threshold = sympy.lambdify(free_syms, pos_thresh[0][m], modules="numpy")

    #%%
    eig_func = sympy.lambdify(free_syms, leading_eig, modules="numpy")
    return eig_func, m_thresh_pos, m_thresh_neg

eig_func, m_thresh_pos, m_thresh_neg = get_analytics()

def jac_and_eigs(s_val, h_val, m_val, alpha_val, c_val=0.9):
    """
    Given (s,h,m,alpha) return numerical eigen values
    """
    args = (s_val, h_val, m_val, alpha_val, c_val) if model == "gd" else (s_val, h_val, m_val, alpha_val)

    # J_num   = np.asarray(J_func(*args), dtype=float)
    eig_num = np.asarray(eig_func(*args), dtype=float)
    # print("Eigval:", eig_num)
    return eig_num

def find_hap_eigs(s1, s2, m_val):
    return m_val / s1 + m_val / s2

def find_mapped(s, c, h, gtype, analytic = False):
    '''
    Old version:
    Read the old mapping result pickle for gd fixation regime 
    and find the mapped ngd
    '''
    mapped = None
    type = gtype
    if analytic: 
        denom = (
        -2 * (c**2) * (h**2) * (s**2)
        + 4 * (c**2) * h * s
        - 2 * (c**2)
        - 2 * c * (h**2) * (s**2)
        + 4 * c * h * s
        - 2 * c
        + s
        )
    
        s_ngd = denom  # from your formula, s_ngd equals this polynomial

        num = c * h * s - c + h * s
        h_ngd = num / denom if denom != 0 else float("nan")

        return s_ngd, h_ngd
    if type == "haploid":
        gridResult = load_pickle(f"mapping_result/h{h}_hap_grid001_G_fix.pickle")
        sMap_grid, wms_grid = gridResult['map'], gridResult['ngC']
        mapped =  sMap_grid[(s, c, h)]
    elif type == "diploid":
        gridResult = load_pickle(f"mapping_result/h{h}_grid_fix001_G.pickle")
        sMap_grid = gridResult
        if (s, c, h) in sMap_grid:
            mapped =  sMap_grid[(s, c, h)]
    return mapped

def plot_haploid(params):
    '''
    Variables: m, alpha, s_val
    Plot a haploid partition for two variables while fixing the third one
    s_val is negative
    '''
    type, model = "haploid", "ngd"
    fixed_vals = {'m': 0.1, 's': params['s'], 'alpha': 2.0}
    m_range, alpha_range = (0, 1.0), (0.1, 2.0) 
    s_range = (-10.0, -0.0)
    resolution = 300
    m_vals = np.linspace(m_range[0], m_range[1], resolution)
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], resolution)
    s_vals = np.linspace(s_range[0], s_range[1], resolution)

    # Fix m, plot parition on alpha vs s_val space
    m_val = fixed_vals['m'] = 0.2
    fixed = "m"
    x, y = "alpha", "s1"
    s2_vals = np.linspace(*s_range, resolution)
    alpha_vals = np.linspace(*alpha_range, resolution)
    A, S2 = np.meshgrid(alpha_vals, s2_vals)

    # Calculate s1 = s2 / -alpha (s1 > 0, s2 < 0)
    S1 = S2 / -A

    # Initialize result array
    gs_partition = np.zeros_like(S1, dtype=int)

    # Loop to evaluate eigenvalue condition
    for i in range(S1.shape[0]):
        for j in range(S1.shape[1]):
            s1 = S1[i, j]
            s2 = S2[i, j]
            if 1+s2 > 1 or 1+s2 < 0:
                continue
            eig = find_hap_eigs(s1, s2, m_val)
            gs_partition[i, j] = int(abs(eig) < 1.0)  # 1 if gene swamping occurs

    stored = {'params': (S1,S2,A), 'partition': gs_partition}
    with open(f"{type}_{model}_gs_parition_{fixed}{fixed_vals[fixed]}.pickle", "wb") as fout:
        pickle.dump(stored, fout)
        
    # Plotting using alpha (x-axis), s1 (y-axis)
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("Blues")
    pcm = plt.pcolormesh(A, S1, gs_partition, cmap=cmap, vmin=0, vmax=1, shading='auto')

    titlestr = f"{fixed}={fixed_vals[fixed]}"
    plt.title(f"{type} {model.upper()} Gene Swamping Partition ({titlestr})")
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    # Get the actual colors from the colormap
    cmap = plt.get_cmap("Blues")
    # plt.colorbar(im, label='Eigenvalue Magnitude', format='%.1f')
    legend_elements = [
        Patch(facecolor=cmap(0), edgecolor='black', label='No Gene Swamping'),
        Patch(facecolor=cmap(1.0), edgecolor='black', label='Gene Swamping')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Gene Swamping Partition")
    plt.savefig(f'plot_partition/{type.lower()}_{model}_partition_{titlestr}.png', dpi=600, bbox_inches='tight')
    plt.show()

def hap_to_gd(sngd):
    with open("../scripts/haploid_gd_all.pickle", "rb") as f:
        hap_gd = pickle.load(f)
    if sngd in hap_gd:
        return hap_gd[sngd][0]
    else:
        return None

def plot_mapped_gd(params, tol=1e-6):
    type, model = "haploid", "ngd"
    fixed = 'm'
    fixed_val = 0.2
    params[fixed] = fixed_val
    # get param dict for gd
    gd_p = params.copy()
    gd_p[fixed] = fixed_val
    gd_p.update({
        'q1': 0.8,          # Initial frequency in population 1
        'q2': 0.2,          # Initial frequency in population 2
        'target_steps': 1000})
    with open(f"{type}_{model}_gs_parition_{fixed}{fixed_val}.pickle", "rb") as f:
        pres = pickle.load(f)
    
    S1, S2, A = pres['params']
    ngd_part = pres['partition']
    gd_part = np.zeros_like(ngd_part, dtype=int)
    for i in range(ngd_part.shape[0]):
        for j in range(ngd_part.shape[1]):
            s1 = S1[i, j]
            s2 = S2[i, j]
            alpha = A[i, j]
            if 1+s2 > 1 or 1+s2 < 0:
                continue
            # s1 -> (s1_gd, c1_gd, h1_gd); s2->(s2_gd, c2_gd, h2_gd)
            gd_mapped1 = hap_to_gd(s1)
            gd_mapped2 = hap_to_gd(s2)
            if gd_mapped1 == None or gd_mapped2 == None:
                continue
            alpha_gd = gd_mapped2[0]/gd_mapped1[0] # gd alpha
            # m is the same as ngd, alpha and config are updated
            gd_p.update({
                's': gd_mapped1[0],
                'c': gd_mapped1[1],
                'h': gd_mapped1[2],
                'alpha': alpha_gd,
            })
            gd_out = gd_simulation(gd_p)
            # gene swamping if both q1 and q2 go to ~0
            gd_part[i, j] = int( abs(gd_out['q1'][-1]) < tol and abs(gd_out['q2'][-1]) < tol )
    # Plotting using alpha (x-axis), s1 (y-axis)
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("Blues")
    pcm = plt.pcolormesh(A, S1, gd_part, cmap=cmap, vmin=0, vmax=1, shading='auto')

    titlestr = f"{fixed}={fixed_val}"
    plt.title(f"{type} {model.upper()} Gene Swamping Partition ({titlestr}) in Mapped GD Model")
    plt.xlabel(f"alpha")
    plt.ylabel(f"s1")
    # Get the actual colors from the colormap
    cmap = plt.get_cmap("Blues")
    # plt.colorbar(im, label='Eigenvalue Magnitude', format='%.1f')
    legend_elements = [
        Patch(facecolor=cmap(0), edgecolor='black', label='No Gene Swamping'),
        Patch(facecolor=cmap(1.0), edgecolor='black', label='Gene Swamping')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Gene Swamping Partition (GD)")
    plt.savefig(f'plot_partition/{type.lower()}_{model}_partition_{titlestr}_mapped.png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_diploid_partition(params, x, y):
    '''
    Parameters:
        params: dict with keys 's', 'h', 'm', 'alpha', 'c'
        x: str, name of the x-axis parameter
        y: str, name of the y-axis parameter
    
    Make a partition based on the leading eigen value for gene drive model
    - Option 1: Fix c, m, and alpha, plot gene swamping partition in s, h space 
    - Option 2: Fix s, c, and h, plot gene swamping partition in m, alpha space

    - If findMap = false: can plot partition figure for haploid/diploid ngd/gd using the leading eigenvalue check
    - If findMap = true: plot the mapped haploid/diploid ngd based on the gd configuration in params

    - If type == diploid: Plot the partition of the (m, alpha) space for a fixed (s, h) 
    - If type == haploid: plot the partition of the (m, alpha) space for a fixed s_val
    '''
    findMap = False
    model = "ngd"
    fixed1, fixed2, fixed3 = "s", "c", "h"
    m_val, alpha_val = params['m'], params['alpha']
    h_val, s_val, c_val = params['h'], params['s'], params['c']
    m_range, alpha_range = (0, 1.0), (0.0, 2.0) 
    s_range,c_range, h_range = (0, 1.0), (0, 1.0), (0, 1.0)
    resolution = 300
    m_vals = np.linspace(m_range[0], m_range[1], resolution)
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], resolution)
    s_vals = np.linspace(s_range[0], s_range[1], resolution)
    c_vals = np.linspace(c_range[0], c_range[1], resolution)
    h_vals = np.linspace(h_range[0], h_range[1], resolution)
    di_eigvals = np.zeros((len(m_vals), len(alpha_vals)))
    di_partition = np.zeros((len(m_vals), len(alpha_vals)))
    hap_eigvals = np.zeros((len(m_vals), len(alpha_vals)))

    # only getting the partition for ngd
    if type == "Haploid" and not findMap:
        for i, m_val in enumerate(m_vals):
            for j, alpha_val in enumerate(alpha_vals):
                args = (s_val, m_val, alpha_val)
                # hap_eigvals[i, j] = np.asarray(eig_func(*args), dtype=float)
                hap_eigvals[i, j] = find_hap_eigs(s_val, -s_val * alpha_val, m_val)
    if type == "Diploid" and not findMap:
        for i, m_val in enumerate(m_vals):
            for j, alpha_val in enumerate(alpha_vals):
                args = (s_val, h_val, m_val, alpha_val, c_val) if model == "gd" else (s_val, h_val, m_val, alpha_val)
                # print("args:::::::::", args)
                di_partition[i, j] = 0 if m_val >= m_thresh_pos(*args) and m_val <= m_thresh_neg(*args) else 1
                # di_eigvals[i, j] = jac_and_eigs(*args) # find leading eigval for diploid gd/ngd
                # print(alpha_val, m_val, m_threshold(*args))
                # eigval = np.asarray(eig_func(s_val, h_val, m_val, alpha_vals), dtype=float)[0]
                # # print(eigval)
                # di_eigvals[i, j] = eigval
    elif findMap:
        # get the partition for mapped ngd model
        hap_eigvals = np.zeros((len(m_vals), len(alpha_vals)))
        hap_partition = np.zeros((len(m_vals), len(alpha_vals)))
        mapped_config = find_mapped(s_val, c_val, h_val)
        for i, m_val in enumerate(m_vals):
            for j, alpha_val in enumerate(alpha_vals):
                if type == "Haploid": # haploid ngd
                    s1 = -mapped_config
                    s2 = s1 * alpha_val
                    # print(s1, s2)
                    args = (s1, s2, m_val)
                    # hap_eigvals[i, j] = find_hap_eigs(*args)
                    # print("mval/s1, alpha")
                    # print(m_val / s1,  alpha_val/abs(1-alpha_val))
                    hap_partition[i, j] = 1 if m_val / s1 > alpha_val/abs(1-alpha_val) else 0

                elif type == "Diploid": #diploid ngd
                    s2, h1 = mapped_config[0], mapped_config[1]
                    s1 = s2 / alpha_val
                    print(s2, s1, h1)
                    di_eigvals[i,j] = jac_and_eigs(s2, h1, m_val, alpha_val)
    
    plt.figure(figsize=(8, 7))
    mapped = "mapped" if findMap else ""
    # print("hap partition", hap_partition)
    if type == "Diploid":
        # print(di_eigvals)
        # di_partition = abs(di_eigvals) < 1.0
        print(di_partition)
        im = plt.imshow(di_partition, aspect="auto", extent=[*alpha_range, *m_range], origin="lower", cmap="Blues", vmin=0, vmax=1)
    else: 
        hap_partition = abs(hap_eigvals) < 1.0
        im = plt.imshow(hap_partition, aspect="auto", extent=[*alpha_range, *m_range], origin="lower", cmap="Blues", vmin=0, vmax=1)
    
    # title string
    if type == "Diploid":
        titlestr = f"{fixed1}={params[fixed1]}, {fixed2}={params[fixed2]}, {fixed3}={params[fixed3]})" if model == "gd" else f"{fixed1}={params[fixed1]}, {fixed3}={params[fixed3]})"
    else: 
        titlestr = f"s={s_val}"
    plt.title(f"{type} {model.upper()} Gene Swamping Partition ({titlestr})")
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    # Get the actual colors from the colormap
    cmap = plt.get_cmap("Blues")
    # plt.colorbar(im, label='Eigenvalue Magnitude', format='%.1f')
    legend_elements = [
        Patch(facecolor=cmap(0), edgecolor='black', label='No Gene Swamping'),
        Patch(facecolor=cmap(1.0), edgecolor='black', label='Gene Swamping')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Gene Swamping Partition")
    plt.savefig(f'plot_partition/{type.lower()}_{model}{mapped}_partition_{titlestr}_2.png', dpi=600, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    params = {"s": -0.5, "h":0.8, "m": 0.01, "alpha": 2.5, "c": 0.8}
    # plot_diploid_partition(params, "alpha", "m")
    # plot_haploid(params)
    # plot_mapped_gd(params)
    # print("\nNumeric Jacobian:\n", Jn)
    # print("Eigen-values:", lams)


# %%
