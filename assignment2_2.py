#!/usr/bin/env python
"""
Compute the exit probability of the voter model on the 2-dimensional square
lattice.
"""

import random
import matplotlib.pyplot as plt
import numpy as np

N=25
SIMULATIONS = 100

def matrix_consensus(matrix):
    consensus = matrix[0,0]
    for i in range(N):
        for j in range(N):
            if (matrix[i,j] != consensus):
                return 0
    return consensus

def run_with_p(p):
    lattice = np.zeros((N,N))
    spin_up = 0
    spin_down = 0
    #  Init with values according to P
    for i in range(N):
        for j in range(N):
            r = random.random()
            if (r < p):
                spin_up += 1
                lattice[i,j] = 1
            else:
                spin_down += 1
                lattice[i,j] = -1

    print('Spin-Ups: %d, Downs: %d' % (spin_up,spin_down))
    #print(lattice)
    consensus = 0
    while(consensus == 0):
        # Get a random site
        x = random.randint(0, N-1)
        y = random.randint(0, N-1)
        neighbours = []
    #    print('Getting neibours for (%d,%d)' % (x,y))
        for i in [x+1,x-1]:
            if i >= 0 and i < N:
                neighbours.append((i,y))
        for j in [y+1,y-1]:
            if j >= 0 and j < N:
                neighbours.append((x,j))

        chosen_neighbour = random.choice(neighbours)
        chosen_value = lattice[chosen_neighbour[0], chosen_neighbour[1]]
        #print('Chose neighbour %d, %d with value %d' % (chosen_neighbour[0], chosen_neighbour[1], chosen_value))

        lattice[x,y] = chosen_value
        consensus = matrix_consensus(lattice)
        #print(lattice)

    print('Consensus reached! %d' % consensus)

    return consensus

p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#p_values = [0.1,0.9]
results = []

for p in p_values:
    print('Spin-up probablity %f' % p)
    total = 0
    for i in range(SIMULATIONS):
        consensus = run_with_p(p)
        print('%d - Consensus %d' % (i, consensus))
        if (consensus == 1):
            total += 1
    average = total / float(SIMULATIONS)
    print('Total: %f Average: %f' % (total, average))
    results.append(average)

plt.plot(p_values, results, 'go')
plt.ylabel('Average number of Spin Up consensus')
plt.xlabel('Probability for Spin Up')
plt.title("Voter model exit probability - %d X %d Lattice" % (N,N))
plt.show()


