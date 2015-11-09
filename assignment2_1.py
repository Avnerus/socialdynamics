#!/usr/bin/env python
"""
Investigate the Deffuant model on a complete graph with N=1200 agents,
"""

from networkx import *
import random
import itertools
import matplotlib.pyplot as plt

N=1200
SIMULATIONS = 100 

def run_with_confidence_bound(confidence_bound):
    G=nx.empty_graph(N)

    # Generate the nodes and edges
    nodes = []
    for i in range(N):
        nodes.append((i,{'opinion': random.random()}))

    edges=itertools.combinations(range(N),2)

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    #print('Running')
    
    biggest_change = 1

    while(biggest_change > 0.000001):
        biggest_change = run_N_times(G, G.edges(), confidence_bound)
        #print('Biggest change in run %f'% biggest_change)

    # Compute clusters
    nodes = G.nodes(True)
    opinions = list(set(n[1]['opinion'] for n in nodes))
    opinions.sort()

    micro_clusters = []
    macro_clusters = []

    current_cluster = []
    cluster_minimum = opinions[0]
    for opinion in opinions:
        #print('Difference: %f: '% (opinion - cluster_minimum))
        if (opinion - cluster_minimum) < 0.0000001:
            current_cluster.append(opinion)
        else:
            # Is it a macro or micro cluster?
            if (len(current_cluster) >= 10):
                macro_clusters.append(current_cluster)
            else:
                micro_clusters.append(current_cluster)
            current_cluster = [opinion]
            cluster_minimum = opinion


    if (len(current_cluster) >= 10):
        macro_clusters.append(current_cluster)
    else:
        micro_clusters.append(current_cluster)

    #print("%d Macro Clusters" % (len(macro_clusters)))
    return len(macro_clusters)

def run_N_times(G,edges,confidence_bound):
    biggest_change = 0

    for i in range (N):
        (index1, index2) = random.choice(edges)
        node1 = G.node[index1]
        node2 = G.node[index2]

        confidence = abs(node1['opinion'] - node2['opinion'])
        #print('confidence between %f and %f is %f' % (node1['opinion'], node2['opinion'], confidence))

        if (confidence <= confidence_bound):
            newOpinion1 = node1['opinion'] + 0.5*(node2['opinion'] - node1['opinion'])
            #print('node %d changed opinion from %f to %f' % (index1, node1['opinion'], newOpinion1))
            newOpinion2 = node2['opinion'] + 0.5*(node1['opinion'] - node2['opinion'])
            #print('node %d changed opinion from %f to %f' % (index2, node2['opinion'], newOpinion2))

            change = abs(newOpinion1 - node1['opinion'])
            if (change > biggest_change):
                biggest_change = change

            node1['opinion'] = newOpinion1
            node2['opinion'] = newOpinion2

    return biggest_change


confidence_bounds = [0.05, 0.1, 0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
#confidence_bounds = [0.45, 0.5]
results = []

for confidence_bound in confidence_bounds:
    print('Confidence bound %f' % confidence_bound)
    total = 0
    for i in range(SIMULATIONS):
        macro_clusters = run_with_confidence_bound(0.5)
        print('%d - %d macro clusters ' % (i, macro_clusters))
        total += macro_clusters
    average = total / float(SIMULATIONS)
    print('Total: %f Average: %f' % (total, average))
    results.append(average)

plt.plot(confidence_bounds, results, 'ro')
plt.ylabel('Average number of Macro Clusters (N >= 10)')
plt.xlabel('Confidence Bound')
plt.title("Deffuant model simulation - %d agents" % N)
plt.show()


