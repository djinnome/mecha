#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 21:46:21 2022

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

from mecha.classes.utils import plot_ternary
#______________________________________________________________________________
          
planting = Mechanism('planting')
current_mechanism = planting

# a stand-in number for the synthetic population
n = 1000 

# the number of planting days under consideration
num_days = 9 

# A simplex that has `num_days` dimensions with uniform (i.e., Dirichlet) distribution over them
date_simplex = Simplex(
    name='date_simplex',
    init_val = rng.dirichlet(np.ones(num_days),n) )

# # the first three points in the simplex
# print(date_simplex.val[:3],"\n")

plot_ternary(vertex_labels = ['a','b','c'], points=date_simplex.val[:,(0,1,2)])


# We need a cohort of those who have planted:
planted = Cohort(
    name='planted',
    n = n, 
    init_val = False,
    update_val = lambda self: np.logical_or(self.val,plant_today.val)
                 )

# Now we need a cohort of people who plant "today".  Sampling from the simplex
# returns an index for each member of the population, where an index of 0 means
# today. 

plant_today = Cohort(
    name = 'planted_today',
    # Those who plant today and haven't already planted
    init_val = (date_simplex.sample() == 0) & (~planted.val),
    update_val = lambda self: (date_simplex.sample() == 0) & (~planted.val)
    )

## Test these:  Repeatedly update plant_today and planted and print their
# respective sizes.  Note that the number who plant_today decreases because
# the pool of those who haven't planted yet decreases

for i in range(20):
    plant_today.update()
    planted.update()
    print(f"plant today: {plant_today.size}, planted: {planted.size}")

print()


# Now we want to bias people to plant earlier or later.  We need a function that 
# produces a left- or right-skewed discrete distribution with as many bins as
# num_days.  The bionomial will serve:
    
def binomial_pdf (n,p):
    fact = np.math.factorial
    def prob (r,n,p):
        m = n-1
        return (fact(m)/(fact(r) * fact(m-r))) * (p**r * (1 - p)**(m-r))
    return np.array([prob(i,n,p) for i in np.arange(n)])


# For example, the binomial_pdf for p = .8 skews the probablity of planting
# over the next num_days to "later", whereas a low p skews to "earlier" and
# p = .5 is symmetric around num_days
    
# print(f"skewed 'later': \n{binomial_pdf(num_days,.8)}\n")
# print(f"skewed 'earlier': \n{binomial_pdf(num_days,.2)}\n")
# print(f"symmetric around mean num_days: \n{binomial_pdf(num_days,.5)}\n")


# Finally we can define influences that drive people to plant earlier
# or later.  Start with one that makes worried people plant later. Make 
# a cohort of worried people

worried = Cohort(
    n=n,
    name='worried',
    init_val = rng.random(n) > .5)


dest = Param(
    name = 'where_the_worried_go', 
    init_val = binomial_pdf(num_days,.8)
    )


wpl = Influence(
    name = 'worried_plant_later',
    mechanism = planting,
    n = n,
    simplex = date_simplex,
    condition = True,
    cohort = worried,
    actionlists = [[(date_simplex,dest,.1)]],
    update_params = True
    )

#wpl.describe()

print()

# d,g=MechMap.make(planting,viz=True)
# display(g)


planting.initialize()

# To test this influence, we must first reset planted and plant_today, then
# repeatedly update the `planting` mechanism.  If the influence is working
# as it should, then the probability mass of "later" days should increase

def prob_later ():
    return np.sum(date_simplex.val[:,0] + date_simplex.val[:,1] + date_simplex.val[:,2]) / n
    
    
planted.reset()
plant_today.reset()




#%%

for i in range(20):
    wpl.run_rule()
    planting.update()
    planted.update()
    plant_today.update()
    early = np.sum(date_simplex.val[:,(0,1,2)],axis=1)
    medium = np.sum(date_simplex.val[:,(3,4,5)],axis=1)
    later = np.sum(date_simplex.val[:,(6,7,8)],axis=1)
    combined = np.stack([early,medium,later]).T
    print (f"number of planted: {planted.size}, prob of planting 'later': {prob_later():.4f} ")
    plot_ternary(vertex_labels = ['a','b','c'], points=combined)


# 

