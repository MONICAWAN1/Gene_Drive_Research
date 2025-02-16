import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def wm(s, n, target_steps, q_init):
    freqs = np.zeros(n)
    freqs[0] = q_init
    for t in range(target_steps-1):
        freqs[t+1] = freqs[t] + s * freqs[t] * (1-freqs[t])
        # freqs[t+1] = freqs[t]**2 * (1-s) / (1-s*(freqs[t]**2))
    return freqs

def run_model(params):
    s = params['s']
    c = params['c']
    h = params['h']
    ts = params['target_steps']
    # s_c = (1 - h * s) * c ## Gametic on modelrxiv
    s_c = (1 - s) * c ## zygotic in paper
    s_n = 0.5 * (1 - h * s) * (1 - c)
    freqs = np.zeros(params['n'])
    freqs[0] = params['q0']
    for t in range(ts-1):
        curr_q = freqs[t]
        w_bar = curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (s_c + 2 * s_n) + (1 - curr_q)**2
        freqs[t+1] = (curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (s_c + s_n)) / w_bar
    return freqs


def linearReg(result1, result2):
    diff1 = diff2 = 0
    q0 = result1
    q0_m = result2
    for i in range(len(q0)):
        diff1 += (q0[i]-q0_m[i])**2
    return diff1**0.5

def mapping(params):
    minVal, maxVal, step = 0.1, 1, 0.1
    s_range = np.arange(minVal, maxVal, step)
    seffMap = dict()
    n = params['n']
    q_init = params['q0']
    configs = []
    s_vals = np.arange(minVal, maxVal, 0.01)
    c_vals = np.arange(0, maxVal, 0.1)
    h_vals = np.arange(0, maxVal, 0.1)
    for s in s_vals:
        for c in c_vals:
            for h in h_vals:
                configs.append((s, 0.8, 0))
    f_out = open('s_map.txt', 'w')
    f_out.write(f"S in well-mixed population\t\tGene Drive Configuration (S_Effective)\n")
    wms = []
    gds = []
    for s_nat in s_range:
        print('s_natural', s_nat)
        best_diff = 10000
        best_conf = None
        gd_results = dict()
        wm_curve = wm(s_nat, params['n'], params['target_steps'], params['q0'])
        for (s, c, h) in configs:
            params['s'], params['c'], params['h'] = s, c, h
            gd_curve = run_model(params)
            diff = linearReg(wm_curve, gd_curve)
            if diff < best_diff:
                best_diff = diff
                best_conf = (round(float(s), 3), round(float(c), 3), round(float(h), 3))
            gd_results[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = gd_curve
        print('best:', best_conf)
        f_out.write(f"{round(float(s_nat), 2)}\t\t\t\t\t\t\t{best_conf}\n")
        seffMap[float(s_nat)] = {best_conf: float(best_diff)}
        wms.append(wm_curve)
        gds.append(gd_results[best_conf])

    time = np.arange(0, params['target_steps'])

    plt.figure(figsize = (15, 7))
    cmap1 = plt.get_cmap('Purples')
    cmap2 = plt.get_cmap('YlOrBr')

    for i in range(len(wms)):
        s = s_range[i]
        wm_curve = wms[i]
        gd_curve = gds[i]
        time = np.arange(0, params['target_steps'])
        w_color = cmap1(1.*i/len(wms))
        g_color = cmap2(1.*i/len(wms))
        plt.plot(time, wm_curve, marker = 'o', color = w_color, linestyle = '-', label = f'well mixed s = {round(s, 2)}')
        plt.plot(time, gd_curve, marker = 'o', color = g_color, linestyle = '-', label = f'gene drive model s = {round(s, 2)}')

    param_text = f"orange: well-mixed population\nblue: population with gene drive\npopulation size = {n}\nq_initial = {q_init}"
    plt.figtext(0.6, 0.2, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    plt.ylabel('Advantageous Allele Frequency')
    plt.xlabel('Time')
    plt.title("Change in Mutant Allele Frequency")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.grid(True)
    plt.legend(title='population condition', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.show()

def main():
    params = {'n': 100, 'target_steps': 100, 'q0': 0.8}
    mapping(params)
        


if __name__ == '__main__':
    main()

