# get fp vs variable plot multilines
import matplotlib.pyplot as plt
import numpy as np
import os

def plot(resultpath):
    plt.figure(figsize=(10, 5))  # Initialize the figure outside the loop
    plt.xlabel('Graph Transitivity')  # Label for the x-axis
    plt.ylabel('Distribution of Outcomes')
    colors = ['blue', 'green', 'red', 'purple']
    categories = ['fix', 'DTE1', 'DTE2', 'loss']
    bar_width = 0.2

    # Initialize dictionaries to store data
    data = dict()
    for category in categories:
        data[category] = {}
    print(data)
    

    for subdir in sorted(os.listdir(resultpath)):
        if os.path.isdir(os.path.join(resultpath, subdir)):
            m_val = subdir.split('m')[-1]
            for resultfile in os.listdir(os.path.join(resultpath, subdir)):
                with open(os.path.join(resultpath, subdir, resultfile), 'r') as file:
                    trials = 0
                    outcomes = {category: 0 for category in categories}
                    phi = None

                    for line in file:
                        if line.startswith('# triangles'):
                            parts = line.strip().split(', ')
                            phi = float(line[:line.find(',')][line.find('=')+2:])
                            k = float(parts[1][parts[1].find('=')+2:])
                            param_dict = {
                            'triangles': phi,
                            'k': k,
                            's': float(parts[2][parts[2].find('=')+2:]),
                            'c': float(parts[3][parts[3].find('=')+2:]),
                            'h': float(parts[4][parts[4].find('=')+2:]),
                            'q0': float(parts[5][parts[5].find('=')+2:]),
                            'target': int(parts[6][parts[6].find('=')+2:]),
                            'm': float(parts[7][parts[7].find('=')+2:]),
                            'repeats': int(parts[8][parts[8].find('=')+2:]),
                            'target_steps': int(parts[9][parts[9].find('=')+2:]),
                            'geneflow': parts[10][parts[10].find('=')+2:],
                        }
                        elif line.strip() and line.strip()[0].isdigit():
                            vals = line.strip().split()
                            trial, q0, q1 = int(vals[0]), float(vals[1]), float(vals[2])
                            trials += 1

                            if q0 > 0.99 and q1 > 0.99:
                                outcomes['fix'] += 1
                            elif q0 < 0.99 and q0 > 0.01:
                                if q0 > q1:
                                    outcomes['DTE1'] += 1
                                else:
                                    outcomes['DTE2'] += 1
                            elif q0 < 0.01 and q1 < 0.01:
                                outcomes['loss'] += 1

                    if trials > 0 and phi is not None:
                        for category in categories:
                            print(category, data)
                            if phi not in data:
                                data[category][phi] = []
                            data[category][phi].append(outcomes[category] / trials)

    print(data)
    for idx, category in enumerate(categories):
        transitivity_values = sorted(data[category].keys())
        means = [np.mean(data[category][trans]) for trans in transitivity_values]
        errs = [np.std(data[category][trans]) for trans in transitivity_values]
        plt.bar(np.array(transitivity_values) + idx * bar_width, means, bar_width, label=category, color=colors[idx], yerr=errs, capsize=5)

    param_text = '\n'.join([f'{key}: {val}' for key, val in param_dict.items() if key != 'triangles' and key != 'target'])
    plt.figtext(0.85, 0.6, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    path = 'PA_results'
    plot(path)

if __name__ == '__main__':
    main()