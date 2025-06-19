# %%
import sympy
import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# %%
q = sympy.Symbol('q')
s = sympy.Symbol('s')
c = sympy.Symbol('c')
h = sympy.Symbol('h')

expr_gd = (q**2 * (1-s) + q * (1-q) * (1+c) * (1 - h*s)) / (q**2 * (1-s) + 2 * q * (1-q) * (1 - h*s) + (1-q)**2)

# %%
q1, q2, q3 = sympy.solvers.solve(expr_gd - q, q)

# solve for s at q3 = 0 and q3 = 1
s_q3_0 = sympy.solvers.solve(q3, s)[0]
s_q3_1 = sympy.solvers.solve(q3 - 1, s)[0]

# find dq and evaluate at q3, if > 1 then unstable 
expr_gd_derivative = sympy.diff(expr_gd, q) 
expr_gd_derivative_q3 = sympy.simplify(expr_gd_derivative.subs(q, q3))

s_q3_0_lambda = sympy.lambdify((c, h), s_q3_0, 'numpy')
s_q3_1_lambda = sympy.lambdify((c, h), s_q3_1, 'numpy')
q3_lambda = sympy.lambdify((s, c, h), q3, 'numpy')
derivative_q3_lambda = sympy.lambdify((s, c, h), expr_gd_derivative_q3, 'numpy')
expr_gd_0_5_lambda = sympy.lambdify((s, c, h), expr_gd.subs(q, 0.5), 'numpy')

# expr_gd_derivative2 = sympy.diff(expr_gd_derivative, q)
# expr_gd_derivative2_0_5 = sympy.simplify(expr_gd_derivative2.subs(q, 0.5))
# derivative2_0_5_lambda = sympy.lambdify((s, c, h), expr_gd_derivative2_0_5, 'numpy')
# derivative2_0_5_mesh = derivative2_0_5_lambda(s_mesh, c_mesh, h_chosen)

# %%
partition_map = {
    0: 'stable',
    1: 'unstable',
    2: 'dq=1',
    3: 'fixation',
    4: 'loss'
}

cmap = plt.cm.get_cmap('Set3')
cmap = ListedColormap(cmap([0,1,2,3,4]))
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # Create boundaries between -1, 0, 1
norm = BoundaryNorm(bounds, cmap.N)

# %%
h_chosen_list = [0.1, 0.3, 0.5, 0.8]

# h_chosen = 0.0
for h_chosen in h_chosen_list:
    c_chosen = np.arange(0.01, 1.0, 0.01)
    s_chosen = np.arange(0.01, 1.0, 0.01)

    c_mesh, s_mesh = np.meshgrid(c_chosen, s_chosen)
    partition = np.zeros_like(c_mesh)
    partition = partition.astype(int)

    q3_mesh = q3_lambda(s_mesh, c_mesh, h_chosen)
    derivative_q3_mesh = derivative_q3_lambda(s_mesh, c_mesh, h_chosen)
    expr_gd_0_5_mesh = expr_gd_0_5_lambda(s_mesh, c_mesh, h_chosen)

    s_q3_0_line = s_q3_0_lambda(c_chosen, h_chosen)
    s_q3_1_line = s_q3_1_lambda(c_chosen, h_chosen)

    # stable 0
    partition[((0 < q3_mesh) * (q3_mesh < 1)) * (derivative_q3_mesh < 1)] = 0

    # unstable 1
    partition[((0 < q3_mesh) * (q3_mesh < 1)) * (derivative_q3_mesh > 1)] = 1

    # dq=1 2
    # partition[((0 < q3_mesh) * (q3_mesh < 1)) * (derivative_q3_mesh == 1)] = 2

    # fixation 3
    partition[((q3_mesh >= 1) + (q3_mesh <= 0)) * (expr_gd_0_5_mesh > 0.5)] = 3

    # loss 4
    partition[((q3_mesh >= 1) + (q3_mesh <= 0)) * (expr_gd_0_5_mesh < 0.5)] = 4

    partition_unique = np.unique(partition)
    # partition_unique = partition_unique.astype(int)

    plt.figure(figsize=(6, 6))
    plt.pcolormesh(c_mesh, s_mesh, partition, cmap=cmap, norm=norm, shading='auto')

    # Overlay the curve
    if h != 0 and h != 0.5:
        plt.plot(c_chosen, s_q3_0_line, color='black', linewidth=1)
    if h != 1:
        plt.plot(c_chosen, s_q3_1_line, color='black', linewidth=1)
    # plt.colorbar(ticks=boundaries)

    legend_elements = [
        Patch(facecolor = col, edgecolor='black', label=partition_map[i_c]) for i_c, col in enumerate(cmap.colors) if i_c in partition_unique
    ]

    plt.legend(handles=legend_elements, title='Regime', loc='upper right')

    # Decorations
    plt.xlabel(r'Conversion Factor, $c$')
    plt.ylabel(r'Selection Coeffiecient, $s$')
    plt.title(f"h = {h_chosen}")
    # plt.legend()
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'figures/GD_partition_h_{h_chosen}.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    # --- NEW: build regimes dict ---
    regimes = {
        'stable':   [],   # internal eq, derivative<1
        'unstable': [],   # internal eq, derivative>1
        'fixation': [],   # drives to 1
        'loss':     []    # drives to 0
    }

    # Walk over every grid cell
    n_rows, n_cols = partition.shape
    for i in range(n_rows):
        for j in range(n_cols):
            code = partition[i, j]
            s_val = s_mesh[i, j]
            c_val = c_mesh[i, j]

            if code == 0:           # stable
                eq = q3_mesh[i, j]
                regimes['stable'].append(((s_val, c_val, h_chosen), eq))

            elif code == 1:         # unstable
                eq = q3_mesh[i, j]
                regimes['unstable'].append(((s_val, c_val, h_chosen), eq))

            elif code == 3:         # fixation
                eq = 1.0
                regimes['fixation'].append(((s_val, c_val, h_chosen), eq))

            elif code == 4:         # loss
                eq = 0.0
                regimes['loss'].append(((s_val, c_val, h_chosen), eq))

    # --- NEW: save to pickle ---
    out_path = f"../pickle/h{h_chosen}_gametic_stability_res.pickle"
    with open(out_path, "wb") as pf:
        pickle.dump(regimes, pf)

    print(f"Saved regimes for h={h_chosen} → {out_path}")

    #%%
    regime = "fixation"
    f_out = open(f"stability_res/h{h}_g_{regime}.txt", 'w') #### change file name!!!!
    f_out.write(f"gene drive model configuration\t\tequilibrium\n")
    for state in regimes.keys():
        if state == regime:  ### change state check
            for config, eq_val in regimes[state]:
                s, c, h = config
                s, c, h = round(float(s), 3), round(float(c), 3), round(float(h), 3)
                f_out.write(f"(s, c, h) = {(s, c, h)}\t\teq = {eq_val}\n")
    
    print(f"Partition written to {f_out}")
    #%%
    # -----------------------------------------------------------------
    # write the dictionary to a text file
    # -----------------------------------------------------------------
    os.makedirs("phase_partition", exist_ok=True)
    h_val = h
    fname = f"phase_partition/partition_h{h_val}.txt"
    with open(fname, "w") as fout:
        for regime in ['stable', 'unstable', 'fixation', 'loss']:
            fout.write(f"## {regime}\n")
            for (cfg, eq) in regimes[regime]:
                s_val, c_val, h_val = cfg
                fout.write(f"s={s_val:.3f}, c={c_val:.3f}, h={h_val:.3f}, eq={eq:.4f}\n")
            fout.write("\n")

    print(f"Partition written to {fname}")

# %%
