#!/usr/bin/env python
# Based on https://github.com/hemmer/comp-phys-tests/blob/master/monte-carlo/pyVicsek/vicsek.py

import numpy as np
#from pylab import *
#import matplotlib.pyplot as plt
import sys

class VicsekModel(object):

    def __init__(self, L=7, eta=2.0, N=300, num_steps=10, simulations=1, r=1.0, dt=1, v=0.03):

        self.L = L              # length of box
        self.halfL = L * 0.5    # used for periodic BCs
        self.N = N          # number of particles
        self.eta = eta      # noise
        self.r2 = r * r     # interaction radius squared
        self.dt = dt        # timestep size
        self.v = v          # magnitude of velocity
        self.num_steps = num_steps
        self.simulations = simulations

        self.velocities = np.zeros((N, 2))
        self.angles = np.zeros(N)
        self.positions = np.zeros((N, 2))

        # first index is for each particle,
        # second for which of its neighbours are in range
        self.particles_in_range = np.eye(N, dtype=np.bool)

    # initialises positions randomly in a box length L,
    # initialises velocites in random directions with mag v
    def initialise_experiment(self):

        # initialise positions
        self.positions = np.random.rand(N, 2) * L

        # generate random directions on [0, 2pi]
        self.angles = np.random.rand(self.N) * 2  * np.pi
        # and use this to find velocity components in x/y directions
        self.velocities[:, 0] = np.cos(self.angles) * self.v  
        self.velocities[:, 1] = np.sin(self.angles) * self.v
        # the components of velocity in x and y direction are v*cos(theta) and v*sin(theta), respectively.

    # this is where the main experiment is carried out
    def main(self, visual_mode):
        print('==Running vicsek model== %d Particles, %d simulations with noise eta %f ' % (self.N, self.simulations, self.eta))

        # set up plotting stuff
        if visual_mode:

            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            wframe = None

            # reset positions
            self.initialise_experiment()
            lastv_a = 0
            diff = 1


            while(diff > 0.0001):

                for step in xrange(self.num_steps):
                    self.perform_step()

                oldcol = wframe
                wframe = self.plot_grid(ax)
                if oldcol:
                    ax.collections.remove(oldcol)

                v_a = self.find_avg_vel()
                diff = v_a - lastv_a
                #print "v_a =", v_a, 'diff: ', diff
                plt.draw()
                lastv_a = v_a

        # otherwise we are in data intensive mode
        else:

            results = []

            for i in range(self.simulations):
                #print('Simulation %d ' % i)

                # reset positions
                self.initialise_experiment()
                lastv_a = 0
                diff = 1

                while(diff > 0.0001):
                    
                    for step in xrange(self.num_steps):
                        self.perform_step()

                    v_a = self.find_avg_vel()
                    diff = v_a - lastv_a
                    #print "v_a =", v_a, 'diff: ', diff
                    lastv_a = v_a


                #print "Stationary at v_a %f" % lastv_a
                results.append(lastv_a)

        mean = np.mean(results)
        std = np.std(results)

        return (mean, std)

    # perform one timestep
    def perform_step(self):

         

        # Choose a random particle
        n = np.random.randint(self.N - 1)
#        n = 0 

        # Move it according to its velocity
        self.positions[n, :] = self.positions[n, :] + self.velocities[n, :] * self.dt

        # apply periodic boundary conditions
        self.positions[n, :] = np.mod(self.positions[n, :], self.L)

        # find which particles are within distance r
        self.find_particles_in_range_for(n)
        near_angles = self.angles[self.particles_in_range[n, :]]
        mean_directions = np.arctan2(np.mean(np.sin(near_angles)), np.mean(np.cos(near_angles)))
        self.angles[n] = mean_directions + ((np.random.random() - 0.5) * self.eta)
        self.velocities[n, 0] = np.cos(self.angles[n]) * self.v
        self.velocities[n, 1] = np.sin(self.angles[n]) * self.v


        """
        # find which particles are within distance r
        self.find_particles_in_range()

        # setup array for saving average directions
        mean_directions = np.zeros(self.N)

        for p in xrange(self.N):
            # get the array of angles for particles in range of particle p
            near_angles = self.angles[self.particles_in_range[p, :]]

            # and average over these
            mean_directions[p] = np.arctan2(np.mean(np.sin(near_angles)),
                                            np.mean(np.cos(near_angles)))

        # new direction is average of surrounding particles + noise
        self.angles = mean_directions + noise_increments

        # using these new directions, we can find the velocity
        # vectors such that all have magnitude v
        self.velocities[:, 0] = np.cos(self.angles) * self.v
        self.velocities[:, 1] = np.sin(self.angles) * self.v
        # the components of velocity in x and y direction are v*cos(theta) and v*sin(theta), respectively.
        """


    def find_particles_in_range_for(self, n):

        # each particle is within its own range, which corresponds
        # to the identity matrix
        self.particles_in_range = np.eye(self.N, dtype=np.bool)

        for q in [x for x in xrange(self.N) if x != n]:

            # difference vector of the 2 points
            diff = self.positions[n, :] - self.positions[q, :]

            # apply minimum image criteria, i.e. use mirror
            # image if closer
            for dim in xrange(2):
                while diff[dim] > self.halfL:
                    diff[dim] = diff[dim] - self.L
                while diff[dim] < -self.halfL:
                    diff[dim] = diff[dim] + self.L


            # get the distance squared to (avoid sqrts)
            dist2 = np.sum(np.power(diff, 2))

            # and see if this is within the allowed range
            in_range = dist2 < (self.r2)
            self.particles_in_range[n, q] = in_range

        #print 'Particles in range of %d: %s' % (n, str(self.particles_in_range[n, :] ))
                
    # Updates the NxN boolean grid of whether particles are in range: first
    # index is for each particle, second for which of its neighbours are in
    # range. As the resulting matrix is symmetric, it doesn't really matter...
    def find_particles_in_range(self):

        # each particle is within its own range, which corresponds
        # to the identity matrix
        self.particles_in_range = np.eye(self.N, dtype=np.bool)

        for p in xrange(self.N):
            for q in xrange(p + 1, self.N):

                # difference vector of the 2 points
                diff = self.positions[p, :] - self.positions[q, :]

                # apply minimum image criteria, i.e. use mirror
                # image if closer
                for dim in xrange(2):
                    while diff[dim] > self.halfL:
                        diff[dim] = diff[dim] - self.L
                    while diff[dim] < -self.halfL:
                        diff[dim] = diff[dim] + self.L


                # get the distance squared to (avoid sqrts)
                dist2 = np.sum(np.power(diff, 2))

                # and see if this is within the allowed range
                in_range = dist2 < (self.r2)
                self.particles_in_range[p, q] = in_range
                self.particles_in_range[q, p] = in_range

    # find the average normalised velocity, v_a
    def find_avg_norm_vel(self):

        mean_v = np.sum(self.velocities, axis=0)
        mag_mean_v = np.power(np.sum(np.power(mean_v, 2)), 0.5)

        return mag_mean_v / (self.N * self.v)

    # plot the swarm using matplotlib
    def find_avg_vel(self):

        mean_v = np.sum(self.velocities, axis=0)
        mag_mean_v = np.power(np.sum(np.power(mean_v, 2)), 0.5)

        return mag_mean_v / (self.N)

    def plot_grid(self, ax):
        plotx = self.positions[:, 0]
        ploty = self.positions[:, 1]
        plotu = self.velocities[:, 0]
        plotv = self.velocities[:, 1]

        return ax.quiver(plotx, ploty, plotu, plotv)

# this is the main experiment
N, L = 400, 5

rho = N / float(L * L)
eta = 4
num_steps = N
simulations = 100 


print "num particles:", N
print "system size:", L
print "density:", rho
print "num steps:", num_steps

eta_values = np.arange(2.71, 2*np.pi, 0.3)
y_values = []
stds = []

for eta in eta_values:

    sim = VicsekModel(L, eta, N, num_steps, simulations)
    if len(sys.argv) > 1:
        print "using visual mode"
        sim.main(True)
    else:
        print "using data intensive mode"
        (mean, std) = sim.main(False)
        print('Mean: %f STD: %f' % (mean, std))
        y_values.append(mean)
        stds.append(std)

print('Y:', y_values)
print('STD: ', stds)

