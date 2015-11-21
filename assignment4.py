#!/usr/bin/env python
"""
Investigate the Axelrod model on a 2-dimensional square lattice
"""

import random
#import matplotlib.pyplot as plt
#from scipy import interpolate   
import networkx as nx
import time
import numpy as np

N = 20
SIMULATIONS = 100
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
    #print len(potential_edges)
    reached_stationary = False
    while not reached_stationary:
        run_cycle(G,potential_edges)
        if (len(potential_edges) == 0):
            reached_stationary = True
        #else:
            #print('Still %d more potential edges '% (len(potential_edges)))

    result = get_cultural_domains(G)    
    print result
    #print potential_edges
    #draw_lattice(G)

    return result['percentage']


def get_potential_edges(G, edges):
    potential_edges = {}
    for edge in edges:
        match_data = get_match_data(G,edge)
        if (match_data['percent'] > 0 and match_data['percent'] < 1):
            potential_edges[edge] = match_data

    return potential_edges


def get_cultural_domains(G):
    domains = {}
    largest = 0
    largest_domain = None

    for n,d in G.nodes_iter(data=True):
        traits = str(d['traits'])
        if not traits in domains:
            domains[traits] = 1
        else:
            domains[traits] += 1
        if domains[traits] > largest:
            largest = domains[traits]
            largest_domain = traits

    return {'largest_domain' : largest_domain, 'percentage': largest / float(N*N)}


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
    cycle = 0
    while (len(potential_edges) > 0 and cycle < CYCLES):
        #print(len(potential_edges))
        edge = random.choice(potential_edges.keys())
        match_data = potential_edges[edge]
        #print(edge, match_data)
        if random.random() < match_data['percent']:
            # Make a change!
            # Random receiver and giver
            receiver = random.randint(0, 1)
            giver = 1 - receiver

            traits1 = G.node[edge[receiver]]['traits']
            traits2 = G.node[edge[giver]]['traits']
            trait_index = random.choice(match_data['no_match'])
            #print('Changing Trait index %d of node %s to be the same as %s!' % (trait_index, str(G.node[edge[0]]), str(G.node[edge[1]])))
            traits1[trait_index] = traits2[trait_index]
            G.node[edge[receiver]]['traits'] = traits1
            #print('Now the traits of %s are %s' % (edge[0], str(G.node[edge[0]])))

            # Update the potential_edges
            changed_edges = nx.edges(G,[edge[receiver]])
            #print('Changed Edges: %s' % str(changed_edges))
            for edge in changed_edges:
                match_data = get_match_data(G, edge)
                is_potential =  (match_data['percent'] > 0 and match_data['percent'] < 1)
                if is_potential and not edge in potential_edges:
                    #print('The edge %s was not in potential edges but now it is! (%s <==> %s)' % (str(edge),str(G.node[edge[0]]['traits']),str(G.node[edge[1]]['traits'])))
                    potential_edges[edge] = match_data
                if not is_potential and edge in potential_edges:
                    #print('The edge %s was in potential edges but now it is not! (%s <==> %s)' % (str(edge),str(G.node[edge[0]]['traits']),str(G.node[edge[1]]['traits'])))

                    potential_edges.pop(edge, None)

        #time.sleep(0.1)
        cycle += 1
                

def draw_lattice(G):
    traits = nx.get_node_attributes(G, 'traits')
    pos=nx.spring_layout(G, iterations=100)

    nx.draw_networkx_nodes(G,pos,node_size=1000)
    nx.draw_networkx_edges(G,pos,width=2)
    nx.draw_networkx_labels(G,pos, labels=traits)

    plt.show()


#q_values = [50]
q_values = range(2, 101)

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


print results
print stds

#plt.plot(p_values, results, 'g-')
#plt.ylabel('Percentage of largest cultural domain')
#plt.xlabel('Q -Number of traits')

# calc the trendline (if it is simply a linear fitting)
#z = np.polyfit(p_values, results, 2)
#p = np.poly1d(z)


#yinterp = interpolate.UnivariateSpline(p_values, results, s=0.1, k=4)(p_values) 

#tck = interpolate.splrep(q_values, results)
#x2 = np.linspace(2,100, 200)
#y2 = interpolate.splev(x2, tck)
#plt.plot(p_values,yinterp,"r-")
#plt.plot(p_values,p(p_values),"g-")

#plt.plot(q_values, results,x2, y2, 'o')
#plt.errorbar(x=q_values, y=results, yerr = stds, fmt = 'o')

#plt.title("Axelrod model- %d X %d Lattice" % (N,N))
#plt.show()
