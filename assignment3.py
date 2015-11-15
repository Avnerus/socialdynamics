#!/usr/bin/env python
"""
Compute the exit probability of the voter model on the 2-dimensional square
lattice.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

N = 30 
SIMULATIONS = 1

NEIGHBORS = np.ones((3,3))
NEIGHBORS[1,1] = 0

neighbour_count = None
spinup_count = None

def run_with_p(p):
    number_of_non_vacant = 0
    number_of_vacant = 0
    number_of_spin_up = 0
    number_of_spin_down = 0

    vacancy_matrix = np.zeros((N,N))
    spin_matrix = np.zeros((N,N))
    non_vacants = []
    vacants = []


    #  Init with inhabitans according to P
    for i in range(N):
        for j in range(N):
            r = random.random()
            if (r < p):
                # It's a vacant site
                vacancy_matrix[i,j] = 0 
                number_of_vacant += 1
                vacants.append((i,j))
            else:
                vacancy_matrix[i,j] = 1 
                number_of_non_vacant += 1
                non_vacants.append((i,j))

    print ("%d non vacant sites :" % len(non_vacants), non_vacants)

    non_vacants_queue = list(non_vacants)
    # Choosing half
    for i in range(len(non_vacants_queue) / 2):
        index = random.randint(0, len(non_vacants_queue) - 1)
        (x,y) = non_vacants_queue.pop(index)
        number_of_spin_up += 1
        spin_matrix[x,y] = 1
    
    # And the rest is spin down
    for (i,j) in non_vacants_queue:
        spin_matrix[i,j] = 0
        number_of_spin_down += 1

    print('Number of vacant sites: %d, Non-vacant: %d. Spin-Ups: %d, Spin Downs: %d' % (number_of_vacant, number_of_non_vacant, number_of_spin_up,number_of_spin_down))
    print(vacancy_matrix)
    print(spin_matrix)

    global neighbour_count
    global spinup_count

    neighbour_count = signal.convolve2d(vacancy_matrix, NEIGHBORS, mode='same')
    spinup_count = signal.convolve2d(spin_matrix, NEIGHBORS, mode='same')


    everyone_are_happy = False
    while not everyone_are_happy:
        run_cycle(vacancy_matrix, spin_matrix, non_vacants, vacants)
        everyone_are_happy = everyone_happy(spin_matrix, non_vacants)
    print('Everyone happy?', everyone_are_happy)
    draw_lattice(spin_matrix, vacancy_matrix)

    

    return 0

def draw_lattice(spin_matrix, vacancy_matrix):
    show_matrix = np.zeros((N,N))
    for i in range(N):
        for (j) in range(N):
            if (vacancy_matrix[i,j] == 1):
                if (spin_matrix[i,j] == 0):
                    show_matrix[i,j] = 0.5
                else:
                    show_matrix[i,j] = 0.7
            else:
                show_matrix[i,j] = 0.2

    plt.imshow(show_matrix, interpolation='none')
    plt.show()

def run_cycle(vacancy_matrix, spin_matrix, non_vacants, vacants):
    global neighbour_count
    global spinup_count

    for k in range(N*N):
        # Pick an occupied site at random
        rand_occupied = random.randrange(len(non_vacants))
        (i,j) = non_vacants[rand_occupied]
        spin  = spin_matrix[i,j]
        happy = is_happy(spin, (i,j))
        #print ('%d,%d is happy?' % (i,j), happy)

        if not happy:
            # Not happy, trying to move
            rand_vacant = random.randrange(len(vacants))
            (newI,newJ) = vacants[rand_vacant]
         #   print('Trying to move to %d,%d' % (newI,newJ))
            happy = is_happy(spin, (newI, newJ))
            if (happy):
                #print('Yay he is happy, moving')
                vacancy_matrix[i,j] = 0
                vacancy_matrix[newI, newJ] = 1
                spin_matrix[i,j] = 0
                spin_matrix[newI, newJ] = spin

                non_vacants.pop(rand_occupied)
                vacants.pop(rand_vacant)
                non_vacants.append((newI,newJ))
                vacants.append((i,j))

                neighbour_count = signal.convolve2d(vacancy_matrix, NEIGHBORS, mode='same')
                spinup_count = signal.convolve2d(spin_matrix, NEIGHBORS, mode='same')



def everyone_happy(spin_matrix, non_vacants):
    happy = True
    index = 0
    while happy and index < len(non_vacants):
        location = non_vacants[index]
        happy = is_happy(spin_matrix[location], location)
        index += 1

    print index
    return happy


def is_happy(spin, location):
    i,j = location
    num_of_neighbours = neighbour_count[i,j]
    num_of_spin_up = spinup_count[i,j]
    num_of_spin_down = num_of_neighbours - num_of_spin_up

    same_spin = num_of_spin_down if spin == 0 else num_of_spin_up

    #print("Checking happiness in %d,%d with spin %d. It has %d neighbors. %d with spinup and %d with spindown, so %d with the same spin" % (i,j,spin, num_of_neighbours, num_of_spin_up, num_of_spin_down, same_spin))


    same_part = 0
    if (num_of_neighbours == 0):
        happy = True
    else:
        same_part = same_spin / float(num_of_neighbours)
        happy = (same_part > 0.5)

    #print('Same part is %f so happy is' % same_part, happy )
    return happy



#p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
p_values = [0.1]

results = []

for p in p_values:
    print('Vacant probablity %f' % p)
    total = 0
    for i in range(SIMULATIONS):
        density = run_with_p(p)
        print('%d - Density %d' % (i, density))
        total += density

    average = total / float(SIMULATIONS)
    print('Total: %f Average: %f' % (total, average))
    results.append(average)

plt.plot(p_values, results, 'go')
plt.ylabel('Average Density')
plt.xlabel('Probability for Vacant Site')
plt.title("Schelling model- %d X %d Lattice" % (N,N))
#plt.show()

