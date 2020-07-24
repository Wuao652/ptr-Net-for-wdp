#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file implements the class WDP (Winner Determination Problem). This class is used for solving a winner determination problem given a finite sample of submitted XOR bids..
WDP has the following functionalities:
    0.CONSTRUCTOR: __init__(self, bids)
       bids = list of numpy nxd arrays representing elicited bundle-value pairs from each bidder. n=number of elicited bids, d = number of items + 1(value for that bundle).
    1.METHOD: initialize_mip(self, verbose=False)
        verbose = boolean, level of == when initializing the MIP for the logger.
        This method initializes the winner determination problem as a MIP.
    2.METHOD: solve_mip(self)
        This method solves the MIP of the winner determination problem and sets the optimal allocation.
    3.METHOD: __repr__(self)
        Echoe on on your python shell when it evaluates an instances of this class.
    4.METHOD: print_optimal_allocation(self)
        This method printes the optimal allocation x_star in a nice way.

See example_Class_WDP_github.py for an example of how to use the class WDP.
"""

# Libs
import numpy as np
import pandas as pd
import logging
# CPLEX: Here, DOcplex is used for solving the deep neural network-based Winner Determination Problem.
import docplex.mp.model as cpx
import time
# documentation
# http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'
# %%


class WDP:
    def __init__(self, bids):
        # list of numpy nxd arrays representing bundle-value pairs from each bidder. 
        # n=number of elicited bids + 1(the number of units for each item)
        # d = number of items + 1(value for that bundle).
        self.bids = bids  
        self.N = len(bids)-1  # number of bidders
        self.M = bids[0].shape[1] - 1  # number of items
        self.Mip = cpx.Model(name="WDP")  # cplex model
        self.z = {}  # decision variables. z(i) = 1 <=> bidder i gets the bundle
        self.x_star = np.zeros((self.N, self.M))  # optimal allocation of the wdp
        self.z_star = np.zeros((1,self.N))

    def initialize_mip(self, verbose=False):

        for i in range(0, self.N):  # over bidders i \in N
            # add decision variables
            self.z.update({(i): self.Mip.binary_var(name="z({})".format(i))}) 
            
        # add intersection constraints of buzndles for z(i)
        # over items m \in M: for each item
        for m in range(0, self.M):
            self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i)]*self.bids[i][0,m] for i in range(0, self.N)) <= self.bids[-1][0,m]), ctname="CT Intersection Item {}".format(m))

        # add objective
        objective = self.Mip.sum(self.z[(i)]*self.bids[i][0,self.M] for i in range(0, self.N))
        self.Mip.maximize(objective)

        if verbose is True:
            for m in range(0, self.Mip.number_of_constraints):
                    logging.debug('({}) %s'.format(m), self.Mip.get_constraint_by_index(m))
        logging.debug('\nMip initialized')

    def solve_mip(self):
        self.Mip.parameters.timelimit.set(300)
        # self.Mip.solve()
        self.Mip.solve(agent='cloud')
        logging.debug(self.Mip.get_solve_status())
        logging.debug(self.Mip.get_solve_details())
        # set the optimal allocation
        for i in range(0, self.N):
            if self.z[(i)].solution_value != 0:
                self.x_star[i, :] = self.z[(i)].solution_value*self.bids[i][0, :-1]
                self.z_star[0, i] = 1
        return self.x_star, self.z_star

    def solve_mip_multi(self, gap):
        MipCpx = self.Mip.get_cplex()
        MipCpx.objective.set_sense(MipCpx.objective.sense.maximize)
        MipCpx.parameters.timelimit.set(300)
        MipCpx.parameters.mip.pool.replace.set(1)
        MipCpx.parameters.mip.pool.relgap.set(gap)
        MipCpx.populate_solution_pool()

        numsol = MipCpx.solution.pool.get_num()
        print("The solution pool contains %d solutions." % numsol)
        mobj = MipCpx.solution.pool.get_mean_objective_value()
        print("The average objective value of the solutions is %.10g." %mobj)

        nb_vars = self.Mip.number_of_variables
        assert nb_vars == self.N
        z_multi = np.zeros((numsol,self.N))
        for i in range(numsol):
            x_i = MipCpx.solution.pool.get_values(i)
            assert len(x_i) == nb_vars
            for k in range(nb_vars):
                if x_i[k] == 1:
                    z_multi[i,k]=1
        return z_multi

    def __repr__(self):
        print('################################ OBJECTIVE ################################')
        try:
            print('Objective Value: ', self.Mip.objective_value, '\n')
        except Exception:
            print("Not yet solved!\n")
        print('############################# SOLVE STATUS ################################')
        print(self.Mip.get_solve_details())
        print(self.Mip.get_statistics(), '\n')
        try:
            print(self.Mip.get_solve_status(), '\n')
        except AttributeError:
            print("Not yet solved!\n")
        print('########################### ALLOCATED BIDDERs ############################')
        try:
            for i in range(0, self.N):
                if self.z[(i)].solution_value != 0:
                    print('z({})='.format(i), int(self.z[(i)].solution_value))
        except Exception:
            print("Not yet solved!\n")
        print('########################### OPT ALLOCATION ###############################')
        self.print_optimal_allocation()
        return(' ')

    def print_optimal_allocation(self):
        D = pd.DataFrame(self.x_star)
        D.columns = ['Item_{}'.format(j) for j in range(1, self.M+1)]
        print(D)
        print('\n')
        Z = pd.DataFrame(self.z_star)
        Z.columns = ['Bid_{}'.format(j) for j in range(1, self.N+1)]
        print(Z)
        print('\nItems allocated:')
        print(D.sum(axis=0))
        


print('WDP Class imported')
