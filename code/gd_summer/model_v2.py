import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import networkx as nx
import argparse
import os
import random


def wm(s, n, target_steps, q_init):
    freqs = np.zeros(n)
    freqs[0] = q_init
    for t in range(target_steps-1):
        freqs[t+1] = freqs[t] + s * freqs[t] * (1-freqs[t])
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
    # print(result1, result2)
    # print(q0, q1, q0_m, q1_m)
    for i in range(len(q0)):
        diff1 += (q0[i]-q0_m[i])**2
        # diff2 += ((1-q0[i])-(1-q0_m[i]))**2
    return diff1**0.5

def mapping(params):
    s_range = np.arange(0.1, 1.1, 0.1)
    seffMap = dict()
    n = params['n']
    q_init = params['q0']
    configs = []
    s_vals = np.arange(0.1, 1.1, 0.1)
    c_vals = np.arange(0, 1.1, 0.1)
    h_vals = np.arange(0, 1.1, 0.1)
    for s in s_vals:
        for c in c_vals:
            for h in h_vals:
                configs.append((s, c, h))
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
        # time = np.arange(0, params['target_steps'])
        # plt.figure(figsize = (15, 7))
        # plt.plot(time, wm_curve, marker = 'o', color = 'orange', linestyle = '-', label = 'well mixed')
        # plt.plot(time, gd_results[best_conf], marker = 'o', color = 'blue', linestyle = '-', label = 'gene drive model')
        # param_text = f"s in well-mixed = {s_nat}\npopulation size = {n}\nq_initial = {q_init}\ngene drive configuration (s,c,h)=\n{best_conf}"
        # plt.figtext(0.6, 0.2, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
        # plt.ylabel('Advantageous Allele Frequency')
        # plt.xlabel('Time')
        # plt.title("Change in Advantageous Mutant Allele Frequency")
        # plt.legend(title = 'Population Condition', bbox_to_anchor=(1, 1.05), loc='upper left')
        # plt.tight_layout(pad=2.0)
        # plt.grid(True)
        # plt.show()
    print(seffMap, s_range)
    
    # how to express s_effs as numerical values
    # s_effs = [seffMap[s_nat].keys()[0] for s_nat in s_range]
    plt.figure(figsize = (15, 7))
    param_text = f"orange: well-mixed population\nblue: population with gene drive\npopulation size = {n}\nq_initial = {q_init}"
    plt.figtext(0.6, 0.2, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    plt.ylabel('Advantageous Allele Frequency')
    plt.xlabel('Time')
    plt.title("Change in Advantageous Mutant Allele Frequency")
    # plt.legend(title = 'Population Condition', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.tight_layout(pad=2.0)
    plt.grid(True)
    time = np.arange(0, params['target_steps'])

    for wm_curve in wms:
        for gd_curve in gds:
            time = np.arange(0, params['target_steps'])
            plt.plot(time, wm_curve, marker = 'o', color = 'orange', linestyle = '-', label = f'well mixed')
            plt.plot(time, gd_curve, marker = 'o', color = 'blue', linestyle = '-', label = 'gene drive model')

    
    plt.show()
    # plt.figure(figsize = (15, 7))
    # plt.plot(s_range, s_effs, marker = 'o', color = 'orange', linestyle = '-')
    # plt.ylabel('Effective Selection Coefficient in Gene Drive')
    # plt.xlabel('Selection Coefficient in well-mixed natural population')
    # plt.show()
    
        


def plot_network(curr, params, G, pos):
    plt.figure(figsize=(11, 7)) 
    ax = plt.gca() 

    lines = curr['edges']

    target = params['target']

    # Plot each line (edge)
    
    for line in lines:
        x_coords = [line[0][0], line[1][0]]
        y_coords = [line[0][1], line[1][1]]
        ax.plot(x_coords, y_coords, 'lightgray') 
    
    all_dots = curr['topology']
    mutant = curr['nodes'][1]
    wild = curr['nodes'][0]
    q_values = curr['q']

    # norm = mcolors.Normalize(vmin=np.min(q_values), vmax=np.max(q_values))
    cmap = plt.get_cmap('OrRd') 
    
    # Plot each node with a color corresponding to its q value
    for i, dot in enumerate(all_dots):
        color = cmap(q_values[i])
        plt.plot(dot[0], dot[1], 'o', color=color)
        if i == target:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[target], node_color=color, node_size=50, edgecolors='black', linewidths=2)
    
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('q value')

    ax.set_title("Population Network")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    param_text = '\n'.join([f'{key}: {val}' for key, val in params.items()])
    plt.figtext(0.86, 0.5, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    plt.grid(True)
    plt.show()

def run_simulation(edgef, params):
    print(f"Running simulation with parameters: {params}")
    print(f"graph:{edgef}")
    results = []
    edges, nodes = generate_graph(edgef) 
    # edges,nodes = generate_rgg_graph(100, 0.15) # same graph for all repeats
    for single in range(params['repeats']):
        # if single >= 90:
        #     print(edgef, edges, params['target'])
        init_deg = np.sum(edges, axis = 1)[params['target']]
        init_step = initial(params, edges, nodes)
        # get the whole trajectory
        result = get_steps(init_step, params, edges)
        q0_single = result['q_target_curve']
        q1_single = result['q_non_curve']

        # results.append((init_deg, q0_single[-1], q1_single[-1]))    
        results.append((init_deg, q0_single, q1_single))                                        

        # plot_network(result, params, edges, nodes)
        # ### drawing the network:
        # if q0_single[-1] > 1 or q1_single[-1] > 1: 
        #     print(f"trial: {single}, target: {params['target']}")
        #     plot_network(result, params, edges, nodes)
        #     break

        params['target'] += 1
    # if '2_deme' in edgef:
    #     print('q0:', results[0][1], 'q1:', results[0][2])
    
    q0_curve = []
    q1_curve = []
    for t in range(params['target_steps']):
        totalq0, totalq1 = 0, 0
        for trial in range(len(results)):
            totalq0 += results[trial][1][t]
            totalq1 += results[trial][2][t]
        avg_q0curve, avg_q1curve = totalq0/params['repeats'], totalq1/params['repeats']
        q0_curve.append(avg_q0curve)
        q1_curve.append(avg_q1curve)
    # if '2_deme' in edgef:
    #     print('q0:', q0_curve, 'q1:', q1_curve)
    # time = np.arange(0, 100)
    # plt.figure(figsize=(15, 7))
    # plt.plot(time, q0_curve, color = 'blue', linestyle = '-', marker = 'o', label = 'q_target')
    # plt.plot(time, q1_curve, color = 'orange', linestyle = '-', marker = 'x', label = 'q_non_target')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency of Gene Drive Allele')
    # plt.title(f"Gene Drive Allele Frequency and time")
    # plt.legend(['q_target', 'q_non_target'])
    # param_text = '\n'.join([f'{key}: {val}' for key, val in params.items()])
    # plt.figtext(0.85, 0.5, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    # plt.grid(True)
    # plt.show()
    return (q0_curve, q1_curve)

def generate_er_graph(n, p):
    graph = np.zeros([n, n])
    idx = np.triu_indices(n, k=1)
    graph.ravel()[np.ravel_multi_index(idx, (n, n))] = np.array(np.random.rand(len(idx[0])) < p, dtype=int)
    G = nx.from_numpy_array(np.matrix(graph))
    pos = nx.kamada_kawai_layout(G)
    nodes = np.array(list(pos.values()))
    return graph + graph.T, nodes

def generate_rgg_graph(n, d):
    nodes = np.random.rand(n, 2)
    graph = np.zeros([n, n])
    idx = np.triu_indices(n, k=1)
    graph.ravel()[np.ravel_multi_index(idx, (n, n))] = np.array([d > ((nodes[i[0]][0] - nodes[i[1]][0])**2 + (nodes[i[0]][1] - nodes[i[1]][1])**2)**0.5 for i in np.array(idx).T], dtype=int)
    return graph + graph.T, nodes

def get_steps(last_step, params, edges):
    ft = 0
    q1 = [last_step['q_target']]
    q2 = [last_step['q_non_target']]
    edgelines = last_step['edges']
    nodes = last_step['nodes']  
    topo = last_step['topology']
    prev_q = last_step['q']
    # f_out = open('output.txt', 'w')

    for t in range(1, params['target_steps'] + 1):
        result = step(params, last_step, t)
        if not result:
            break
        last_step = result
        ft = t
        target, non_target = last_step['q_target'], last_step['q_non_target']

        ## printing the frequency > 1 case

        # if target > 1 or non_target > 1:
        # info = ',\n'.join([f'{key} = {value}' for key, value in last_step.items() if key not in {'edges', 'topology', 'nodes_q'}])
        # f_out.write(f"target_deme: {q1}, non_target_demes: {q2}, {info}\n")
        # f_out.write(f"target: {params['target']}\n")

        q1.append(target)
        q2.append(non_target)

        prev_q = last_step['q']


    return {'q': prev_q, 'q_target_curve': q1, 'q_non_curve': q2, 'ft': ft, 'target': params['target'], 'nodes': nodes, 'topology': topo, 'edges': edgelines}

def initial(params, edges, nodes):
    np.random.seed()
    M = migration_network(params, edges) 

    q = np.zeros(M.shape[0]) 

    q[params['target']] = params['q0'] 
    nodes_q = np.insert(nodes, 2, np.maximum(q, 0.1), axis=1) if len(nodes) > 0 else np.ndarray([0,3]) 

    # nodes: nodes classified into fix and loss
    return {'q': q, 'q_target': params['q0'], 'q_non_target': 0, 'spillover': 0, 'M': M, 'topology': nodes, 'nodes': [nodes_q[nodes_q[:,2] < 0.5], nodes_q[nodes_q[:,2] >= 0.5]], 'edges': edge_lines_from_nodes(M, nodes)}

# for plotting the network
def edge_lines_from_nodes(M, nodes):
    if len(nodes) == 0:
        return []
    indices = np.triu_indices(M.shape[0], k=1) # M is the adjacency matrix 
    edges = np.array(indices).T[M[indices] > 0.0001] # select only the (i, j) that has significant connection in M
    lines = nodes[edges]
    return lines

def step(params, _step, t):
    if t == 0:
        return initial(params)
    M = np.array(_step['M']) ## Migration network, generated in first step by "initial"
    s = params['s']
    c = params['c']
    h = params['h']
    # s_c = (1 - h * s) * c ## Gametic on modelrxiv
    s_c = (1 - s) * c ## zygotic in paper
    s_n = 0.5 * (1 - h * s) * (1 - c)
    q_tilde = np.dot(_step['q'], M)
    # print(t, q_tilde)
    w_bar = q_tilde**2 * (1 - s) + 2 * q_tilde * (1 - q_tilde) * (s_c + 2 * s_n) + (1 - q_tilde)**2
    q_tag = (q_tilde**2 * (1 - s) + 2 * q_tilde * (1 - q_tilde) * (s_c + s_n)) / w_bar
    nodes_q = np.insert(_step['topology'], 2, np.maximum(q_tag, 0.1), axis=1) if len(_step['topology']) > 0 else np.ndarray([0,3])
    edgelines = _step['edges']


    return {'q': q_tag, 'q_target': q_tag[params['target']], 'q_non_target': (np.sum(q_tag) - q_tag[params['target']]) / (M.shape[0] - 1), 'spillover': len(q_tag[q_tag > 0.5]) - 1, 'M': M, 'topology': _step['topology'], 'nodes': [nodes_q[nodes_q[:,2] < 0.5], nodes_q[nodes_q[:,2] >= 0.5]], 'edges': edgelines}

def generate_graph(filename):
    n = 100
    graph = [[0] * n for _ in range(n)]
    with open(filename, 'r') as file: 
        for line in file:
            edge = line.strip().split(' ')
            # print(edge)
            i, j = int(edge[0]), int(edge[1])
            graph[i][j] = 1
            graph[j][i] = 1
    graph = np.array(graph)
    # print(graph)

    G = nx.from_numpy_array(np.matrix(graph))
    pos = nx.spring_layout(G)
    nodes = np.array(list(pos.values()))
    return graph, nodes

def migration_network(params, graph):
    if params['geneflow'] == 'Fixed_out':
        n = graph.shape[0]
        M = params['m'] * graph / np.maximum(1, np.sum(graph, axis=1))[:,None] 
    elif params['geneflow'] == 'Fixed_in':
        n = graph.shape[0]
        M = params['m'] * graph / np.maximum(1, np.sum(graph, axis=0))
    else:
        n = graph.shape[0]
        M = graph * params['m'] ## All gene flow is equal to m
    
    M[range(n), range(n)] = 1 - np.sum(M, axis=1)
    M = M / np.sum(M, axis=0)
    # print(M)
    return M


def read_parameters(filename):
    parameters = []
    edgefiles = []
    resultfiles = []
    # get amp factors
    ampFactors = dict()
    with open('amp_factor.txt', 'r') as file:
        for line in file:
            if line.strip():
                graphName, ampFact = line.split()
                ampFactors[graphName] = ampFact
    with open(filename, 'r') as file:
        # open parameters.in
        for line in file:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                edges = parts[0] # get the graph file
                G = nx.read_edgelist(edges)
                phi = nx.transitivity(G)
                k = sum([d for (n, d) in nx.degree(G)]) / float(G.number_of_nodes())
                # amp = ampFactors[edges[edges.find('PA_mixpatt'):edges.find('.txt')]]
                edgefiles.append(edges)
                resultfile = parts[1]
                resultfiles.append(resultfile)
                param_dict = {
                    # 'triangles': phi,
                    # 'k': k,
                    # 'amp': float(amp),
                    's': float(parts[2]),
                    'c': float(parts[3]),
                    'h': float(parts[4]),
                    'q0': float(parts[5]),
                    'target': int(parts[6]),
                    'm': float(parts[7]),
                    'repeats': int(parts[8]),
                    'target_steps': int(parts[9]),
                    'geneflow': parts[10],
                }
                parameters.append(param_dict)
    return edgefiles, resultfiles, parameters

def main():
    params = {'n': 100, 'target_steps': 100, 'q0': 0.5}
    mapping(params)
        


if __name__ == '__main__':
    main()

