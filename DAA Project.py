#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import random

def rnd_graph(n, p):
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G[i][j] = 1
                G[j][i] = 1
    return G

def opt_coloring(G):
    n = len(G)
    coloring = [-1] * n
    nodes = list(range(n))
    nodes.sort(key=lambda node: sum(G[node]), reverse=True)
    num_colors = 0

    for node in nodes:
        avail_colors = set(range(num_colors))
        for neighbor in range(n):
            if G[node][neighbor] and coloring[neighbor] in avail_colors:
                avail_colors.remove(coloring[neighbor])
        if not avail_colors:
            coloring[node] = num_colors
            num_colors += 1
        else:
            coloring[node] = min(avail_colors)

    return coloring

n = 20
p = 0.5
graph = rnd_graph(n, p)
optimal_coloring = opt_coloring(graph)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_dim = n
hidden_dim = 128
output_dim = n

input_weights = np.random.rand(input_dim, hidden_dim)
hidden_weights = np.random.rand(hidden_dim, output_dim)
input_bias = np.zeros((1, hidden_dim))
output_bias = np.zeros((1, output_dim))

input_data = np.random.rand(100, n, input_dim)
target_data = np.random.randint(0, output_dim, size=(100, n))

lr = 0.001
epochs = 100

for epoch in range(epochs):
    total_loss = 0

    for sample in range(input_data.shape[0]):
        inp_layer = np.dot(input_data[sample], input_weights) + input_bias
        hid_layer = sigmoid(inp_layer)
        out_layer = np.dot(hid_layer, hidden_weights) + output_bias

        loss = np.mean(np.square(out_layer - target_data[sample]))
        total_loss += loss

        d_out = 2 * (out_layer - target_data[sample]) / out_layer.shape[0]
        d_hid = np.dot(d_out, hidden_weights.T) * sigmoid_derivative(hid_layer)

        hidden_weights -= np.dot(hid_layer.T, d_out) * lr
        input_weights -= np.dot(input_data[sample].T, d_hid) * lr
        input_bias -= np.sum(d_hid, axis=0) * lr
        output_bias -= np.sum(d_out, axis=0) * lr

# Color Correction Phase
def correct_coloring(G, coloring):
    for u, v in G.edges():
        if coloring[u] == coloring[v]:
            avail_colors = set(coloring.values())
            for color in range(len(avail_colors) + 1):
                if color not in avail_colors:
                    coloring[v] = color
                    break
    return coloring

def count_conflicts(G, coloring):
    conflicts = 0
    for u, v in G.edges():
        if coloring[u] == coloring[v]:
            conflicts += 1
    return conflicts

def plot_colored_graph(G, coloring):
    n = len(G)
    pos = {node: (random.random(), random.random()) for node in range(n)}
    colors = [coloring[node] for node in range(n)]

    plt.figure(figsize=(8, 8))

    for u in range(n):
        for v in range(u + 1, n):
            if G[u][v]:
                plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='black', linewidth=2)

    for node in range(n):
        circle = plt.Circle(pos[node], 0.05, color=f'C{colors[node]}', edgecolor='k', zorder=2)
        plt.gca().add_patch(circle)
        plt.text(pos[node][0], pos[node][1], str(node), ha='center', va='center', color='white' if colors[node] == 0 else 'black', zorder=3)

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.show()

plot_colored_graph(graph, optimal_coloring)

def get_colored_nodes(G, coloring):
    colored_nodes = {node: coloring[node] for node in range(len(G))}
    return colored_nodes

colored_nodes_dict = get_colored_nodes(graph, optimal_coloring)
print(colored_nodes_dict)


# In[ ]:




