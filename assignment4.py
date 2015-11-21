#!/usr/bin/env python
"""
Investigate the Axelrod model on a 2-dimensional square lattice
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

N = 4
SIMULATIONS = 1
F = 2
CYCLES = 400

def run_with_q(q):
    G = nx.grid_2d_graph(N,N)
#    draw_lattice(G)

    # Generate traits
    traits = {}
    for node in G:
        trait_values = []
        for i in range(F):        
            trait_values.append(random.randint(1,q))
        traits[node] = trait_values

    nx.set_node_attributes(G, 'traits', traits)
    edges = G.edges()
    potential_edges = get_potential_edges(G, edges)
    print potential_edges
    reached_stationary = False
    while not reached_stationary:
        run_cycle(G,potential_edges)
        if (len(potential_edges) == 0):
            reached_stationary = True
        else:
            print('Still %d more potential edges '% (len(potential_edges)))
            draw_lattice(G)

    
    print potential_edges

    return 0


def get_potential_edges(G, edges):
    potential_edges = {}
    for edge in edges:
        match_data = get_match_data(G,edge)
        if (match_data['percent'] > 0 and match_data['percent'] < 1):
            potential_edges[edge] = match_data

    return potential_edges

def get_match_data(G, edge):
    #print('Checking for any match in edge %s' % str(edge))
    traits1 = G.node[edge[0]]['traits']
    traits2 = G.node[edge[1]]['traits']

    #print('Traits1: %s, Traits2: %s' % (traits1, traits2))

    match = 0
    do_not_match = []
    for i in range(F):
        if traits1[i] == traits2[i]:
            match += 1
        else:
            do_not_match.append(i)

    return {'percent': match / float(F), 'no_match': do_not_match}

def run_cycle(G,potential_edges):
    for cycle in range(CYCLES):
        edge = random.choice(potential_edges.keys())
        match_data = potential_edges[edge]
        #print(edge, match_data)
        if random.random() < match_data['percent']:
            # Make a change!
            traits1 = G.node[edge[0]]['traits']
            traits2 = G.node[edge[1]]['traits']
            trait_index = random.choice(match_data['no_match'])
            #print('Changing Trait index %d of node %s to be the same as %s!' % (trait_index, str(G.node[edge[0]]), str(G.node[edge[1]])))
            traits1[trait_index] = traits2[trait_index]
            G.node[edge[0]]['traits'] = traits1
           # print('Now the traits of %s are %s' % (edge[0], str(G.node[edge[0]])))

            # Update the potential_edges
            changed_edges = nx.edges(G,[edge[0]])
            #print('Changed Edges: %s' % str(changed_edges))
            for edge in changed_edges:
                match_data = get_match_data(G, edge)
                is_potential =  (match_data['percent'] > 0 and match_data['percent'] < 1)
                if is_potential and not edge in potential_edges:
                    #print('The edge %s was not in potential edges but now it is!' % str(edge))
                    potential_edges[edge] = match_data
                if not is_potential and edge in potential_edges:
                    #print('The edge %s was not in potential edges but now it is not!' % str(edge))
                    potential_edges.pop(edge, None)
                

def draw_lattice(G):
    traits = nx.get_node_attributes(G, 'traits')
    pos=nx.spring_layout(G, iterations=100)

    nx.draw_networkx_nodes(G,pos,node_size=1000)
    nx.draw_networkx_edges(G,pos,width=2)
    nx.draw_networkx_labels(G,pos, labels=traits)

    plt.show()


#p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
q_values = [2]
#p_values = [0.2]

results = []
stds = []

for q in q_values:
    print('Q=%d' % q)
    simulations = []
    for i in range(SIMULATIONS):
        domain_fraction = run_with_q(q)
        simulations.append(domain_fraction)
        print('%d - Domain Fraction %f' % (i, domain_fraction))

    mean = np.mean(simulations)
    std = np.std(simulations)
    print('Mean: %f Std: %f' % (mean, std))
    results.append(mean)
    stds.append(std)

#plt.plot(p_values, results, 'g-')
plt.ylabel('Percentage of largest cultural domain')
plt.xlabel('Number of traits')

# calc the trendline (if it is simply a linear fitting)
#z = np.polyfit(p_values, results, 2)
#p = np.poly1d(z)


#yinterp = interpolate.UnivariateSpline(p_values, results, s=0.1, k=4)(p_values) 

#tck = interpolate.splrep(p_values, results)
#x2 = np.linspace(0.1,0.9, 200)
#y2 = interpolate.splev(x2, tck)
#plt.plot(p_values,yinterp,"r-")
#plt.plot(p_values,p(p_values),"g-")

plt.plot(q_values, results, 'o')
plt.errorbar(x=q_values, y=results, yerr = stds, fmt = 'o')

plt.title("Axelrod model- %d X %d Lattice" % (N,N))
#plt.show()
