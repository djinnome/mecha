#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:24:05 2022

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

planting2 = Mechanism('planting2')
current_mechanism = planting2

# a stand-in number for the synthetic population
n = 100

# the number of planting days under consideration
num_days = 9 

# define a counter that counts which day it is
day = Counter()

# A simplex that has `num_days` dimensions with uniform (i.e., Dirichlet) distribution over them
date_simplex = Simplex(
    name='date_simplex',
    init_val = rng.dirichlet(np.ones(num_days),n) )



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
    update_val = lambda self: (date_simplex.sample() == day.val) & (~planted.val)
    )



def binomial_pdf (n,p):
    fact = np.math.factorial
    def prob (r,n,p):
        m = n-1
        return (fact(m)/(fact(r) * fact(m-r))) * (p**r * (1 - p)**(m-r))
    return np.array([prob(i,n,p) for i in np.arange(n)])



worried = Cohort(
    n=n,
    name='worried',
    init_val = rng.random(n) > .5,
    update_val = lambda self: np.logical_and(self.val,~planted.val)
    )



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


deadline_fear = Param(
    name='deadline_fear',
    init_val = binomial_pdf(num_days,.5),
    update_val = lambda self: binomial_pdf(num_days, max((.5 - (day.val/10)),.1))
    )

dpe = Influence(
    name = 'deadline_plant_earlier',
    mechanism = planting,
    n = n,
    simplex = date_simplex,
    condition = True,
    cohort = worried,
    actionlists = [[(date_simplex,deadline_fear.val,.1)]],
    update_params = True
    )



planting.initialize()


def prob_later ():
    return np.sum(date_simplex.val[:,6] + date_simplex.val[:,7] + date_simplex.val[:,8]) / (n - planted.size)





#%%

date_simplex.reset()    
day.reset()
planted.reset()
plant_today.reset()
deadline_fear.reset()



for i in range(10):
    wpl.run_rule()
    dpe.run_rule()
    day.update()
    deadline_fear.update()
    planting.update()
    planted.update()
    plant_today.update()
    worried.update()
    early = np.sum(date_simplex.val[:,(0,1,2)],axis=1)
    medium = np.sum(date_simplex.val[:,(3,4,5)],axis=1)
    later = np.sum(date_simplex.val[:,(6,7,8)],axis=1)
    combined = np.stack([early,medium,later]).T
    print (f"day:  {day.val}, num planted: {planted.size}, prob of planting 'later': {prob_later():.4f} ")
    plot_ternary(vertex_labels = ['a','b','c'], points=combined, color_by = planted.val)





