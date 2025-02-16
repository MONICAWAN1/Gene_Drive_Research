import numpy as np
import matplotlib.pyplot as plt
import math

# def isclose(a, b, rel_tol=1e-19, abs_tol=0.0):
#     return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def defaults():
    return {
        's': 0.5,
        'c': 1,
        'h': 1,
        'q0': 1,
        'target': 0,
        'm': 0.01,
        'geneflow': 'Scale with degree',
        'graph': 'RGG',
        'n': 100,
        'p': 0.15,
        'd': 0.15,
        'sf_m': 2,
        'rand_seed': '',
        'repeats': 1,
        'target_steps': 100
    }

def generate_er_graph(n, p):
    graph = np.zeros([n, n])
    idx = np.triu_indices(n, k=1)
    graph.ravel()[np.ravel_multi_index(idx, (n, n))] = np.array(np.random.rand(len(idx[0])) < p, dtype=int)
    return graph + graph.T, np.array([]) # why is this empty?

def generate_rgg_graph(n, d):
    nodes = np.random.rand(n, 2)
    graph = np.zeros([n, n])
    idx = np.triu_indices(n, k=1)
    graph.ravel()[np.ravel_multi_index(idx, (n, n))] = np.array([d > ((nodes[i[0]][0] - nodes[i[1]][0])**2 + (nodes[i[0]][1] - nodes[i[1]][1])**2)**0.5 for i in np.array(idx).T], dtype=int)
    return graph + graph.T, nodes

def generate_scalefree_graph(n, m=2):
    edges = np.zeros([n,n])
    edges[1,0] = 1
    for i in range(m, n):
        prob = (np.sum(edges, axis=1) + np.sum(edges, axis=0)) / (2 * np.sum(edges))
        edges[i,np.random.choice(n, p=prob, size=m)] = 1
    return edges + edges.T, np.array([])

def generate_graph(params):
    if params['graph'] == 'ER':
        return generate_er_graph(params['n'], params['p'])
    elif params['graph'] == 'RGG':
        return generate_rgg_graph(params['n'], params['d'])
    elif params['graph'] == 'scalefree':
        return generate_scalefree_graph(params['n'], params['sf_m'])

def migration_network(params, graph):
    if params['geneflow'] == 'Fixed out':
        n = graph.shape[0]
        M = graph * (params['m'] / np.maximum(1, np.sum(graph, axis=0))) ## If gene flow out is fixed, columns should sum to 1 before normalization
        M[range(n), range(n)] = 1 - params['m'] ## Diagonal is simply 1 - m because the gene flow out is fixed
        M = M / np.sum(M, axis=1)[:,None] ## Normalize rows
    elif params['geneflow'] == 'Fixed in':
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

def initial(params):
    if params['rand_seed'] != '': # what is rand_seed: a seed for generating the random numbers
        np.random.seed(int(params['rand_seed']))
    else:
        np.random.seed() # get a random value for the seed 
    edges, nodes = generate_graph(params) # edge: adjacency matrix; node: the topology - what is it?  
    M = migration_network(params, edges) # M is the modified edges matrix
    q = np.zeros(M.shape[0])  # an array representing q at each node 

    degrees = np.sum(edges, axis=1)
    max_degree = np.argmax(degrees)
    min_degree = np.argmin(degrees)

    # if params['graph'] == 'scalefree':
    #     params['target'] = np.random.choice(M.shape[0]) if params['target'] == 0 else params['target']
    # else:
    if params['target'] == 0:
        params['target'] = np.random.choice(M.shape[0]) 
    elif params['target'] == 'max':
        params['target'] = max_degree
    elif params['target'] == 'min':
        params['target'] = min_degree
    q[params['target']] = params['q0']  # what is q0? -- initial frequency in target population (which is a single random dot)
    nodes_q = np.insert(nodes, 2, np.maximum(q, 0.1), axis=1) if len(nodes) > 0 else np.ndarray([0,3])  # nodes is the array of coordinates for each node
    return {'q': q, 'q_target': params['q0'], 'q_non_target': 0, 'spillover': 0, 'M': M, 'topology': nodes, 'nodes': [nodes_q[nodes_q[:,2] < 0.5], nodes_q[nodes_q[:,2] >= 0.5]], 'edges': edge_lines_from_nodes(M, nodes)}

def step(params, _step, t):
    if t == 0:
        return initial(params)
    M = np.array(_step['M']) ## Migration network, generated in first step by "initial"
    s = params['s']
    c = params['c']
    h = params['h']
    s_c = (1 - h * s) * c ## Gametic
    s_n = 0.5 * (1 - h * s) * (1 - c)
    q_tilde = np.dot(_step['q'], M)
    w_bar = q_tilde**2 * (1 - s) + 2 * q_tilde * (1 - q_tilde) * (s_c + 2 * s_n) + (1 - q_tilde)**2
    q_tag = (q_tilde**2 * (1 - s) + 2 * q_tilde * (1 - q_tilde) * (s_c + s_n)) / w_bar
    nodes_q = np.insert(_step['topology'], 2, np.maximum(q_tag, 0.1), axis=1) if len(_step['topology']) > 0 else np.ndarray([0,3])
    return {'q': q_tag, 'q_target': q_tag[params['target']], 'q_non_target': (np.sum(q_tag) - q_tag[params['target']]) / (M.shape[0] - 1), 'spillover': len(q_tag[q_tag > 0.5]) - 1, 'M': M, 'topology': _step['topology'], 'nodes': [nodes_q[nodes_q[:,2] < 0.5], nodes_q[nodes_q[:,2] >= 0.5]], 'edges': edge_lines_from_nodes(M, np.array(_step['topology']))}

def single_run(params):
    last_step = None
    for t in range(0, params['target_steps'] + 1):
        result = step(params, last_step, t)
        if not result:
            break
        last_step = result
    return last_step

def run(params, **kwargs):
    stats = []
    for last_step in [single_run(params) for i in range(params['repeats'])]:
        result = {'spillover': last_step['spillover'] / (params['n'] - 1)}
        stats.append(result)
    return {'repeats': stats}

def plot_network(params):
    plt.figure(figsize=(10, 8)) 
    ax = plt.gca()  # Get the current axes
    curr = initial(params)

    print(curr)

    lines = curr['edges']

    # Plot each line (edge)
    
    for line in lines:
        x_coords = [line[0][0], line[1][0]]
        y_coords = [line[0][1], line[1][1]]
        ax.plot(x_coords, y_coords, 'b-') 
    
    all_dots = curr['topology']
    mutant = curr['nodes'][0]
    wild = curr['nodes'][1]
    print(wild)
    for dot in all_dots:
        if any(np.array_equal(dot, highlight[:-1]) for highlight in wild): 
            plt.plot(dot[0], dot[1], 'o', color='orange')
        else: 
            plt.plot(dot[0], dot[1], 'o', color='blue')
    
    ax.set_title("Network Graph")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.grid(True)  # Enable grid for better visibility
    plt.show()

# params = defaults()
# print(initial(params))
# plot_network(params)


# def spillover_m():
#     params = defaults()

#     # print(single_run(params))
#     # return 

#     y_data = []
#     x_data = np.arange(0, 1.01, 0.01)
#     params['repeats'] = 100
    
#     for m in x_data:
#         params['m'] = m
#         stats = run(params)['repeats']
#         spill_sum = 0

#         for spill in stats:
#             spill_sum += spill['spillover']

#         spill_avg = spill_sum / len(stats)
#         y_data.append(spill_avg)

#     fig1 = plt.figure()
#     plt.plot(x_data, y_data, label='Spillover vs. Migration Rate')
#     plt.xlabel('Migration Rate (m)')
#     plt.ylabel('Average Spillover')
#     plt.title('Spillover as a Function of Migration Rate')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     return fig1

# def run_freqs(params):
#     q1 = []
#     q2 = []
#     for last_step in [single_run(params) for i in range(params['repeats'])]:
#         q1.append(last_step['q_target'])
#         q2.append(last_step['q_non_target'])
#     return (q1, q2)

### get the trajectory ##########################
def get_steps(params):
    last_step = None
    q1 = []
    q2 = []
    ft = 0
    for t in range(0, params['target_steps'] + 1):
        result = step(params, last_step, t)
        if not result:
            break
        last_step = result
        ft = t
        q1.append(last_step['q_target'])
        q2.append(last_step['q_non_target'])
        # Fixation time
        # if len(q1) >= 2 and len(q2) >= 2 and q1[-1] == q1[-2] and q2[-1] == q2[-2]:
        #     break

    return {'q_target_curve': q1, 'q_non_curve': q2, 'ft': ft}


### fixation probability #####################
def count_fix(params):
    last_step = None
    target_fix = non_fix = 0
    q1_prev = q2_prev = None

    rel_tolerance = 1e-02
    abs_tolerance = 1e-20

    for t in range(0, params['target_steps'] + 1):
        result = step(params, last_step, t)
        if not result:
            break
        last_step = result

        q1 = last_step['q_target']
        q2 = last_step['q_non_target']
        
        # print('q1', q1, 'q2', q2)
        # print('q1_prev', q1_prev, 'q2_prev', q2_prev)

        if q1_prev != None and q2_prev != None:
            if (math.isclose(q1, q1_prev, rel_tol= rel_tolerance, abs_tol=abs_tolerance) 
                and math.isclose(q2, q2_prev, rel_tol=rel_tolerance, abs_tol=abs_tolerance)):
                # print('stable', t, q1, q1_prev, q2, q2_prev)
                if q1 > 0.999:
                    target_fix = 1
                if q2 > 0.999:
                    non_fix = 1
                return {'target': target_fix, 'non': non_fix}
        
        q1_prev, q2_prev = q1, q2

    print('check count:', t, format(q1_prev, '.30f'), format(q1, '30f'), format(q2_prev, '.30f'), format(q2, '.30f'))
    return {'target': target_fix, 'non': non_fix}


def fp_single(params):
    results = [count_fix(params) for _ in range(params['repeats'])]
    q1_prob = sum(result['target'] for result in results) / params['repeats']
    q2_prob = sum(result['non'] for result in results) / params['repeats']

    return (q1_prob, q2_prob)

def fp_all(params, var, runs):
    params['target_steps'] = 400
    params['repeats'] = 50
    params['target'] = 'max'
    params['h'] = 1
    params['m'] = 0.01
    params['graph'] = 'RGG'

    v_data = np.arange(0, 1, 0.01)  # range of s
    q1_fp = []
    q2_fp = []
    for v in v_data:
        params[var] = v
        
        results = [fp_single(params) for _ in range(runs)]
        q1_avg = np.mean([result[0] for result in results])
        q2_avg = np.mean([result[1] for result in results])

        print(v, q1_avg, q2_avg)

        q1_fp.append(q1_avg)
        q2_fp.append(q2_avg)
    
    
    plt.figure(figsize=(15, 7))
    plt.plot(v_data, q1_fp, color = 'blue', linestyle = '-', marker = 'o', label = 'q_target average fp')
    plt.plot(v_data, q2_fp, color = 'orange', linestyle = '-', marker = 'x', label = 'q_non_target average fp')
    plt.xlabel(var)
    plt.ylabel('probability of fixation')
    plt.title(f"Fixation Probability and {var} (choose max degree node as target)")
    plt.legend(['q_target', 'q_non_target'])
    plt.grid(True)

    param_text = '\n'.join([f'{key}: {val}' for key, val in params.items() if key != var])
    plt.figtext(0.85, 0.5, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))

    plt.show()

params = defaults()
# params['target_steps'] = 1500
# params['repeats'] = 1
# params['graph'] = 'ER'
# params['s'] = 0.6
# params['target'] = 'min'
# print(fp_single(params))
# print(math.isclose(4.732857581648554e-146, 4.732857581648553e-146, rel_tol = 1e-09))
fp_all(params, 's', 10)

### q_t plots based on a single varying parameter ##############################
def q_figure(v):
    # q_target and q_non_target versus time 
    params = defaults()
    params['target_steps'] = 400
    time_steps = np.arange(0, params['target_steps']+1, 1)
    params['repeats'] = 50
    params['m'] = v
    params['s'] = 0
    params['h'] = 1
    params['target'] = 'min'
    params['graph'] = 'RGG'
    plt.figure(figsize=(15, 7))
    
    # for each repeat
    for single in range(params['repeats']):
        
        # get the whole trajectory
        q1_single = get_steps(params)['q_target_curve']
        q2_single = get_steps(params)['q_non_curve']

        
        plt.plot(time_steps, q1_single, color = 'blue', lw = 0.5, linestyle='-')
        plt.plot(time_steps, q2_single, color = 'orange', lw = 0.5, linestyle='-')

    plt.title(f"Gene Drive Allele Frequencies Over Time (min degree node as target)")
    plt.xlabel('Time Step')
    plt.ylabel('q')
    plt.legend(['q_target', 'q_non_target'])
    plt.grid(True)

    param_text = '\n'.join([f'{key}: {val}' for key, val in params.items()])
    plt.figtext(0.85, 0.6, f"Parameters:\n{param_text}", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))

    plt.show()

def mult_q_fig(vals):
    for v in vals:
        q_figure(v)

### fixation time versus selectoin coefficient ################################
def fixation_time(params):
    # average fixation_time of 100 repeats in a single run for an s_val
    all_times = [get_steps(params)['ft'] for _ in range(params['repeats'])]

    avg_ft = sum(all_times)/len(all_times)

    return avg_ft

def fixation():
    params = defaults()
    params['target_steps'] = 1500
    # time_steps = np.arange(0, params['target_steps']+1, 1)
    params['repeats'] = 50
    params['graph'] = 'RGG'

    x_data = np.arange(0, 1.0, 0.005)  # range of s
    ft_data = []

    for s in x_data:
        params['s'] = s
        # print(fixation_time(params))
        # continue
        max_s = max(fixation_time(params) for _ in range(5))
        # for _ in range(5):
        # 	ft = fixation_time(params) # avg fixation time for all repeats for an s
        # 	if ft > max_s:
        # 		max_s = ft
        print(s, 'max', max_s)
        ft_data.append(max_s)

    # return
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, ft_data, label='Fixation Time vs. s')
    plt.xlabel('s')
    plt.ylabel('time of fixation')
    plt.title('Fixation Time and s')
    plt.legend()
    plt.grid(True)
    plt.show()

### plot states ############################################################
def get_state(params):
    last_step = None
    q1 = []
    q2 = []
    ft = 0
    for t in range(0, params['target_steps'] + 1):
        result = step(params, last_step, t)
        if not result:
            break
        last_step = result
        q1.append(last_step['q_target'])
        q2.append(last_step['q_non_target'])
        if len(q1) >= 2 and len(q2) >= 2 and q1[-1] == q1[-2] and q2[-1] == q2[-2]:
            q1_last, q2_last = q1[-1], q2[-1]
            print(q1_last, q2_last)
            if q1_last > q2_last:
                return "DTE"
            elif q1_last > 0.5 and q2_last > 0.5:
                return "fixation"
            elif q1_last < 0.5 and q2_last < 0.5:
                return "loss"
            break

def run_states(s):
    params = defaults()
    params['target_steps'] = 2000
    params['repeats'] = 100
    params['s'] = s
    params['graph'] = 'ER'
    states = [get_state(params) for _ in range(params['repeats'])]
    print(s, states)
    if states.count('DTE') >= states.count("fixation") and states.count('DTE') >= states.count("loss"):
        return 0.5
    elif states.count('fixation') >= states.count("DTE") and states.count('fixation') >= states.count("loss"):
        return 1
    elif states.count('loss') >= states.count("DTE") and states.count('loss') >= states.count("fixation"):
        return 0
        


s_values = np.linspace(0.4, 0.6, 20)
# states = [run_states(s) for s in s_values]

def plot_states(s_values, states):
    plt.figure(figsize=(10, 5))


    plt.scatter(s_values, states, c=states, cmap='viridis', edgecolor='k')  # Color by state
    plt.yticks([0, 0.5, 1], ["Global Loss", "DTE", "Global Fixation"])
    plt.xlabel('Selection Coefficient (s)')
    plt.ylabel('State')
    plt.title('State of Allele Frequency Against Selection Coefficient')
    plt.grid(True)
    plt.show()

# q_t(0)
# q_t(0.001)
# q_figure(0.05)
# q_figure(0.2)


# m_vals = [0.03 for _ in range(3)]
# mult_q_t(m_vals)

# fixation()

# plot_states(s_values, states)

# allele freqs over time, 10 repeats, modify s
# allele freqs over time, 50 repeats, modify topography
# network structure graph, with points labeled by colors according to q? 
# how is q determined? - target is a single point, others are all non-target points
