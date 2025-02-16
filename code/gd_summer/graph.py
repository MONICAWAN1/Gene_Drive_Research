# generate a graph
import networkx as nx
import matplotlib.pyplot as plt
import os

def generate_er_graph(n, p):
    G = nx.erdos_renyi_graph(n, p)
    return G

def save_graph_to_file(G, file_path):
    with open(file_path, 'w') as f:
        for edge in G.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

def load_and_draw_graph(file_path):
    G = nx.read_edgelist(file_path, nodetype=int)
    nx.draw(G, with_labels=True)
    plt.show()

# Parameters
n = 100  # Number of nodes
p = 0.15  # Probability of an edge

# Generate and save the graph
def generateG(directory):
    numG = 100

    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(numG):
        G = nx.random_geometric_graph(n, p)
        filePath = os.path.join(directory, f"rgg_n100_p015_{i}.txt")
        save_graph_to_file(G, filePath)

generateG('RGG')
