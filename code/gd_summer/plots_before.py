# get fp vs variable plot multilines
import matplotlib.pyplot as plt
import numpy as np
import os

def plot(resultpath):
    plt.figure(figsize=(10, 5))  # Initialize the figure outside the loop
    plt.xlabel('Graph Transitivity')  # Label for the x-axis
    plt.ylabel('Fixation Probability')  # Label for the y-axis
    markers = ['o', 'o', 'o', 'o', 'o']  # Different markers for different lines
    colors = ['b', 'g', 'r', 'c', 'm']  # Different colors for different lines
    marker_idx = 0
    

    for subdir in sorted(os.listdir(resultpath)):
        if os.path.isdir(os.path.join(resultpath, subdir)):
            data = dict()
            m_val = subdir.split('m')[-1]
            for resultfile in os.listdir(os.path.join(resultpath, subdir)):
                with open(os.path.join(resultpath, subdir, resultfile), 'r') as file:
                    trials = 0
                    fix = DTE1 = DTE2 = loss = 0
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
                                fix += 1

                            elif q0 < 0.99 and q0 > 0.01:
                                if q0 > q1:
                                    DTE1 += 1
                                elif q0 < q1:
                                    DTE2 += 1
                            elif q0 < 0.01 and q1 < 0.01:
                                loss += 1
                        
                        if trials > 0: 
                            data[phi] = loss/trials
            x_data = sorted(data.keys())
            y_data = [data[x] for x in x_data]

            # Plot each set of data with unique marker and color
            plt.plot(x_data, y_data, marker=markers[marker_idx % len(markers)], linestyle='-', color=colors[marker_idx % len(colors)], label=f'm={m_val}')
            marker_idx += 1

            print(x_data, y_data)
    plt.legend(['q = 0.5', 'q = 0.8', 'q = 1'])
    plt.grid(True)  
    plt.show()

    # plt.figure(figsize=(10, 5))  # Set the figure size (optional)
    # plt.plot(x_data, y_data, marker='o', linestyle='-', color='b')  # Plot x vs y
    # plt.title('Fixation Probability and graph transitivity on k-degree graphs')  # Title of the plot
    # plt.xlabel('Graph Transitivity')  # Label for the x-axis
    # plt.ylabel('Fixation Probability')  # Label for the y-axis

    param_text = '\n'.join([f'{key}: {val}' for key, val in param_dict.items() if key != 'triangles' and key != 'target'])
    plt.figtext(0.85, 0.6, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))

def single_amp_plot(resultpath):
    outcome = 'Stays in Target Node'
    # outcome = 'Spread to Entire Network'
    # outcome = 'Global Fixation'
    # outcome = 'Global Loss'
    plt.figure(figsize=(12, 6))  # Initialize the figure outside the loop
    plt.xlabel('Target Node Degree')  # Label for the x-axis
    plt.ylabel(f'Number of trials ending up with {outcome}')  # Label for the y-axis
    plt.title(f'{outcome} on PA Networks with Varying Amplification Factors')
    cmap = plt.get_cmap('viridis')
    markers = ['o'] * 100  # Different markers for different lines
    numColors = 100
    values = np.linspace(0, 100, 100) 
    colors = []
    for i in range(numColors):
        color = cmap(1.*i/numColors)
        colors.append(color)
    marker_idx = 0
    ampList = []

    ampDict = dict()
    for resultfile in os.listdir(resultpath):
        with open(os.path.join(resultpath, resultfile), 'r') as file:
            for line in file:
                if line.startswith('# amp'):
                    parts = line.strip().split(', ')
                    amp = float(line[:line.find(',')][line.find('=')+2:])
                    ampDict[amp] = resultfile
    allItems = []
    for key, val in ampDict.items():
        allItems.append((key, val))
    allItems.sort()
    allItems.reverse()
    print(allItems)

    for i in range(100):
        amp, resfile = allItems[i]
        with open(os.path.join(resultpath, resfile), 'r') as file:
            data = dict()
            degCount = {}
            countLossTable = [0] * 100
            countFixTable = [0] * 100
            countStayTable = [0] * 100
            countSpreadTable = [0] * 100
            trials = 0
            for line in file:
                if line.startswith('# amp'):
                    parts = line.strip().split(', ')
                    amp = float(line[:line.find(',')][line.find('=')+2:])
                    # phi = float(line[:line.find(',')][line.find('=')+2:])
                    # k = float(parts[1][parts[1].find('=')+2:])
                    i = 1
                    param_dict = {
                    # 'triangles': phi,
                    # 'k': k,
                    'amp': amp,
                    's': float(parts[i][parts[i].find('=')+2:]),
                    'c': float(parts[i+1][parts[i+1].find('=')+2:]),
                    'h': float(parts[i+2][parts[i+2].find('=')+2:]),
                    'q0': float(parts[i+3][parts[i+3].find('=')+2:]),
                    'target': int(parts[i+4][parts[i+4].find('=')+2:]),
                    'm': float(parts[i+5][parts[i+5].find('=')+2:]),
                    'repeats': int(parts[i+6][parts[i+6].find('=')+2:]),
                    'target_steps': int(parts[i+7][parts[i+7].find('=')+2:]),
                    'geneflow': parts[i+8][parts[i+8].find('=')+2:],
                }
                elif line.strip() and line.strip()[0].isdigit():
                    vals = line.strip().split()
                    trial, deg, q0, q1 = int(vals[0]), int(vals[1]), float(vals[2]), float(vals[3])
                    trials += 1
                    degCount[deg] = degCount.get(deg, 0) + 1
                    # stay: q0 < 0.01, q1 > 0.01
                    # spread: q0 and q1 > 0.01 but one of them < 0.99
                    # fix: q0 and q1 > 0.99
                    # loss q0 and q1 < 0.01

                    if q0 < 0.01 and q1 < 0.01:
                        countLossTable[deg] += 1
                    elif q0 > 0.99 and q1 > 0.99:
                            countFixTable[deg] += 1
                    elif q0 > 0.01 and q1 < 0.01:
                            countStayTable[deg] += 1
                    elif q0 > 0.01: 
                        countSpreadTable[deg] += 1
            for deg in degCount:  
                # print(deg, countTable[deg], degCount[deg])  
                # print(sum(degCount.values()))
                data[deg] = [countFixTable[deg], countStayTable[deg], countLossTable[deg], countSpreadTable[deg]]
            degrees = sorted(data.keys())
            fixProb = [data[x] for x in degrees]
            
            fixes = [fixProb[i][0] for i in range(len(degrees))]
            stays = [fixProb[i][1] for i in range(len(degrees))]
            losses = [fixProb[i][2] for i in range(len(degrees))]
            spread = [fixProb[i][3] for i in range(len(degrees))]
            # print(param_dict['amp'], stays)


            # Plot each set of data with unique marker and color
            if stays[3] != 0:
                plt.plot(degrees, stays, marker=markers[marker_idx % len(markers)], linestyle='-', color=colors[marker_idx % len(colors)], label=f'amp = {amp}')
            else:
                plt.plot(degrees, stays, marker=markers[marker_idx % len(markers)], linestyle='-', color=colors[marker_idx % len(colors)])
            marker_idx += 1
            ampList.append(amp)
            # print(x_data, y_data)

    # plt.legend([f"amp = {a}" for a in ampList])

    # plt.legend(title='Amplication Factor', bbox_to_anchor=(1.05, 1), loc='upper left')
    param_text = '\n'.join([f'{key}: {val}' for key, val in param_dict.items() if key != 'amp' and key != 'target'])
    plt.figtext(0.6, 0.6, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    plt.grid(True)  
    plt.legend(title='Amplification Factor', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.subplots_adjust(right=0.75)  # Adjust subplot to fit the legend
    plt.tight_layout(pad=2.0)
    plt.show()

def histogram(resultpath):
    data = dict()
    degCount = {}
    countLossTable = [0] * 100
    countFixTable = [0] * 100
    countStayTable = [0] * 100
    countSpreadTable = [0] * 100
    ampDict = dict()
    for resultfile in os.listdir(resultpath):
        with open(os.path.join(resultpath, resultfile), 'r') as file:
            for line in file:
                if line.startswith('# amp'):
                    parts = line.strip().split(', ')
                    amp = float(line[:line.find(',')][line.find('=')+2:])
                    ampDict[amp] = resultfile
    allItems = []
    for key, val in ampDict.items():
        allItems.append((key, val))
    allItems.sort()
    print(allItems)

    for amp, resfile in allItems:
        with open(os.path.join(resultpath, resfile), 'r') as file:
            for line in file:
                if line.startswith('# amp'):
                    parts = line.strip().split(', ')        
                    i = 1
                    param_dict = {
                    's': float(parts[i][parts[i].find('=')+2:]),
                    'c': float(parts[i+1][parts[i+1].find('=')+2:]),
                    'h': float(parts[i+2][parts[i+2].find('=')+2:]),
                    'q0': float(parts[i+3][parts[i+3].find('=')+2:]),
                    'target': int(parts[i+4][parts[i+4].find('=')+2:]),
                    'm': float(parts[i+5][parts[i+5].find('=')+2:]),
                    'repeats': int(parts[i+6][parts[i+6].find('=')+2:]),
                    'target_steps': int(parts[i+7][parts[i+7].find('=')+2:]),
                    'geneflow': parts[i+8][parts[i+8].find('=')+2:],
                }
                elif line.strip() and line.strip()[0].isdigit():
                    vals = line.strip().split()
                    trial, deg, q0, q1 = int(vals[0]), int(vals[1]), float(vals[2]), float(vals[3])
                    degCount[deg] = degCount.get(deg, 0) + 1
                    if q0 < 0.01 and q1 < 0.01:
                        countLossTable[deg] += 1
                    elif q0 > 0.99 and q1 > 0.99:
                            countFixTable[deg] += 1
                    elif q0 > 0.01 and q1 < 0.01:
                            countStayTable[deg] += 1
                    elif q0 > 0.01: 
                        countSpreadTable[deg] += 1
    # print(degCount)
    for deg in degCount:  
        # print(deg, countTable[deg], degCount[deg])  
        # print(sum(degCount.values()))
        totalCount = degCount[deg]
        data[deg] = [countFixTable[deg], countStayTable[deg], countLossTable[deg], countSpreadTable[deg]]
    
    degrees = sorted(data.keys())
    fixProb = [data[x] for x in degrees]
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.4
    
    fixes = [fixProb[i][0] for i in range(len(degrees))]
    stays = [fixProb[i][1] for i in range(len(degrees))]
    losses = [fixProb[i][2] for i in range(len(degrees))]
    spread = [fixProb[i][3] for i in range(len(degrees))]

    ax.bar(degrees, fixes, width=bar_width, label='Global Fix', color='#8BBD8A')
    ax.bar(degrees, stays, width=bar_width, bottom=fixes, label='Stay in Initial Population', color='#FFC56E')
    ax.bar(degrees, losses, width=bar_width, bottom=np.array(fixes) + np.array(stays), label='Global Loss', color='#BDB5B6')
    ax.bar(degrees, spread, width=bar_width, bottom=np.array(fixes) + np.array(stays) + np.array(losses), label='Spread to Entire Population', color='#F19797')

    ax.set_xlabel('Target Population Degree')
    ax.set_ylabel('Counts of Outcomes')
    ax.set_title('Final States on PA Graphs')
    ax.legend()

    param_text = '\n'.join([f'{key}: {val}' for key, val in param_dict.items() if key != 'amp' and key != 'target'])
    plt.figtext(0.85, 0.3, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    plt.grid(True)  
    plt.show()


def main():
    path = 'PA_s0.56_m0.005'
    # histogram(path)
    single_amp_plot(path)

if __name__ == '__main__':
    main()