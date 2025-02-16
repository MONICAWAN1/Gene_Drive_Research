### just plotting the network
### network determined by the params file if not random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import networkx as nx
import argparse
import os

# given the location and q final of each node, color the network

def generate_er_graph(n, p):
    graph = np.zeros([n, n])
    idx = np.triu_indices(n, k=1)
    graph.ravel()[np.ravel_multi_index(idx, (n, n))] = np.array(np.random.rand(len(idx[0])) < p, dtype=int)
    G = nx.from_numpy_array(np.matrix(graph))
    pos = nx.spring_layout(G)
    nodes = np.array(list(pos.values()))
    return graph + graph.T, nodes

def k_degree_graph(filename):
    # G = nx.read_edgelist(filename)
    # graph = nx.adjacency_matrix(G)
    # graph = graph.toarray()
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
        M = graph * (params['m'] / np.maximum(1, np.sum(graph, axis=0))) ## If gene flow out is fixed, columns should sum to 1 before normalization
        M[range(n), range(n)] = 1 - params['m'] ## Diagonal is simply 1 - m because the gene flow out is fixed
        M = M / np.sum(M, axis=1)[:,None] ## Normalize rows
    elif params['geneflow'] == 'Fixed_in':
        n = graph.shape[0]
        M = graph * (params['m'] / np.maximum(1, np.sum(graph, axis=1)))[:,None] ## If gene flow in is fixed, row should sum to 1 before normalization
        M[range(n), range(n)] = 1 - np.sum(M, axis=1)
    else:
        n = graph.shape[0]
        M = graph * params['m'] ## All gene flow is equal to m
        M[range(n), range(n)] = 1 - np.sum(M, axis=1) ## Diagonal is 1 - m * connections
    return M

def edge_lines_from_nodes(M, nodes):
    if len(nodes) == 0:
        return []
    indices = np.triu_indices(M.shape[0], k=1) # M is the adjacency matrix 
    edges = np.array(indices).T[M[indices] > 0.0001] # select only the (i, j) that has significant connection in M
    lines = nodes[edges]
    return lines

def initial(params, edges, nodes):
    np.random.seed() # get a random value for the seed 
     # edge: adjacency matrix; node: the topology
    M = migration_network(params, edges) # M is the modified edges matrix

    q = np.zeros(M.shape[0])  # an array representing q at each node 

    q[params['target']] = params['q0']  # q0: initial frequency in target population (which is a single random dot)
    nodes_q = np.insert(nodes, 2, np.maximum(q, 0.1), axis=1) if len(nodes) > 0 else np.ndarray([0,3])  # nodes is the array of coordinates for each node

    # nodes: nodes classified into fix and loss
    return {'q': q, 'q_target': params['q0'], 'q_non_target': 0, 'spillover': 0, 'M': M, 'topology': nodes, 'nodes': [nodes_q[nodes_q[:,2] < 0.5], nodes_q[nodes_q[:,2] >= 0.5]], 'edges': edge_lines_from_nodes(M, nodes)}

def plot_network(curr, params):
    plt.figure(figsize=(10, 8)) 
    ax = plt.gca()  # Get the current axes

    lines = curr['edges']

    target = params['target']

    # Plot each line (edge)
    print(lines)
    
    for line in lines:
        x_coords = [line[0][0], line[1][0]]
        y_coords = [line[0][1], line[1][1]]
        ax.plot(x_coords, y_coords, 'lightgray') 
    
    all_dots = curr['topology']
    mutant = curr['nodes'][1]
    wild = curr['nodes'][0]
    q_values = curr['q']

    norm = mcolors.Normalize(vmin=min(q_values), vmax=max(q_values))
    cmap = plt.get_cmap('OrRd')  # Can choose any colormap that suits your data
    
    # Plot each node with a color corresponding to its q value
    for i, dot in enumerate(all_dots):
        color = cmap(norm(q_values[i]))
        plt.plot(dot[0], dot[1], 'o', color=color)
        if i == target:
            plt.text(dot[0], dot[1], ' Target', color='red', fontsize=12, ha='right')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('q value')

    ax.set_title(f"Population Network ({params['n']} demes)")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    param_text = '\n'.join([f'{key}: {val}' for key, val in params.items()])
    plt.figtext(0.86, 0.5, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
    plt.grid(True)  # Enable grid for better visibility
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Run simulations with parameters from a file.")
    parser.add_argument('param_file', type=str, help="File containing the simulation parameters.")
    return parser.parse_args()

args = parse_args()

def read_parameters(filename):
    parameters = []
    edgefiles = []
    resultfiles = []
    with open(filename, 'r') as file:
        # open parameters.in
        for line in file:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split()
                edges = parts[0]
                G = nx.read_edgelist(edges)
                phi = nx.transitivity(G)
                k = sum([d for (n, d) in nx.degree(G)]) / float(G.number_of_nodes())
                edgefiles.append(edges)
                resultfile = parts[1]
                resultfiles.append(resultfile)
                param_dict = {
                    'n': G.number_of_nodes(),
                    'triangles': phi,
                    'k': k,
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
    args = parse_args()
    edgefiles, resultfiles, params_list = read_parameters(args.param_file)
    # edges, nodes = generate_er_graph(100, 0.15)
    for plotNum in range(len(edgefiles)):
        params = params_list[0]
        edgef = edgefiles[0]
        edges, nodes = k_degree_graph(edgef)
        init_step = initial(params, edges, nodes)
        plot_network(init_step, params)


if __name__ == '__main__':
    main()




