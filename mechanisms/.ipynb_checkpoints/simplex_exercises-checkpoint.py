#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:53:04 2022

@author: prcohen
"""

import os
os.chdir("/Users/zuck016/Projects/Proposals/ASKEM")

import sys
import pandas as pd
import numpy as np
from numpy.random import default_rng
rng = default_rng(1108) # stuff for random sampling; note random seed is set

from mecha.classes.mechanism_classes import Mechanism, Sim, MechMap
from mecha.classes.rule_classes import Rule, Influence
from mecha.classes.param_classes import Param, Cohort, Probs, Counter, Switch, Cycle, Array, Simplex

import mecha.classes.rule_classes as rule_classes
import mecha.classes.param_classes as pc

from mecha.classes.utils import plot_ternary, plot_series, plot_multiseries


'''
Exercise 1:  On each time step, each member of the population can act or not
with probabilities p and q = 1 - p.  Define a population that has a low average
probability of acting, say 0.1 and create a simplex for the population.  


Suppose acting is an absorbing state, meaning that once a member of the population
has acted, they cannot act again and cannot 'go back' to the state of not having
acted.  Examples include planting a crop and voting.  

Run a simulation of N steps in which, at each step, you sample from the simplex
to get the members of the population who have acted. Sample only those who have
not yet acted.  Generate a time series of length N of the total number who have 
acted.
    
    
Exercise 2:  Write an influence that changes the probability of acting.  It should
implement social pressure to act; that is, the probability of acting should increase
as more people act.  Alternatively, the rate associated with the influence might be
proportional to the fraction of the population that has acted. This influence should
apply only to those who have not yet acted.  Run a simulation of N steps and
plot the number of the population that have acted. Consider also plotting the number
who act on each step, so you can see whether this number accelerates.


Exercise 3: Write an influence that increases the probability of acting (or the
rate) as a deadline approaches.  Assume the same deadline for everyone in the
population. Run a simulation of N steps and plot the number of the population 
that have acted. Also plot the number who act on each step, so you can see 
whether this number accelerates.

Exercise 4: Run both influences together, plot the results.

Exercise 5: Consider a third option -- quitting -- in addition to not acting and
acting.  In other words, consider a three-dimensional simplex where the corners
are "have not acted", "have acted" and "quit".  Only those who have not acted
are subject to influences.  Write an influence that increases the probability of
quitting as a deadline approaches. Run a simulation of N steps and plot the 
number of the population  that have acted and the number who have quit. 

'''

acting = Mechanism(name='acting')
current_mechanism = acting

# a stand-in number for the synthetic population
n = 10000


# make an array with a low probability of acting
p = rng.random(n) / 5
q = 1-p

# make a simplex 
s = Simplex(
    name = 's',
    init_val = np.stack([p,q]).T
    )

#keep track of those who have acted
acted = Cohort(
    n = n,
    init_val = False,
    update_val = lambda self: np.logical_or(self.val, s.sample() == 0)
    )

# record a time series
series = []

for i in range(20):
    acted.update()
    series.append(acted.size)
    
plot_series(series)
plot_series(np.diff(series))   


#_____________________________________________________________________________

s.reset()
acted.reset()

def p_cohort (cohort):
    p = cohort.size/cohort.n
    return np.array([p,1-p])
    
social_pressure = Influence(
    name = 'social_pressure',
    n = n,
    simplex = s,
    condition = True,
    actionlists = [[(s, np.array([1,0]), .1 )], 
                   [] ],
    probs = Probs(
        init_val = p_cohort(acted),
        update_val = lambda self: p_cohort(acted)
        ),
    update_params = True
    )

s0,s1 = [],[]

for i in range(20):
    acted.update()
    s0.append(acted.size)

s.reset()
acted.reset()
    
for i in range(20):
    social_pressure.run_rule()
    acted.update()
    social_pressure.probs.update()
    social_pressure.multinomial.update()
    s1.append(acted.size)
    
plot_series(s0,s1)


#%%

# x = rng.random(7)
# p = np.stack([x,1-x]).T
# print(f"{np.sum(p,axis=0)}, {np.sum(r,axis=0)}")
# p = neighbor_max (p)
# print(f"{np.sum(p,axis=0)}, {np.sum(r,axis=0)}")





# def neighbor_max (x,col=0,k=1):
#     '''Given a 2D array, x, this rolls the values by k postions columnswise, giving
#     array y.  Then it takes the maximum row of x and y according to the specified
#     column'''
#     r = np.roll(x,-1,axis=0)
#     z = x[:,0] < r[:,0]
#     x[z]=r[z]
#     return x

# dest = Array(
#     init_val = s.val,
#     update_val = lambda self: neighbor_max(self.val)
#     )


# series=[]
# for i in range(20):
#     dest.update()
#     series.append(np.sum(dest.val,axis=0)[0])
# plot_series(series)
    









