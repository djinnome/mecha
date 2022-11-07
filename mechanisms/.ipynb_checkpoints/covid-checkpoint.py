#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 08:35:24 2022

@author: prcohen
"""
import os
os.chdir("/Users/zuck016/Projects/Proposals/ASKEM")

import numpy as np
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling

import pandas as pd
from mecha.classes.utils import plot_series, plot_multiseries,plot_replicates

from mecha.classes.registry import Registry
from mecha.classes.param_classes import Param, Column, Probs, Cohort
from mecha.classes.mechanism_classes import Mechanism,Sim, MechMap
from mecha.classes.rule_classes import Rule

import mecha.classes.rule_classes as rule_classes
import mecha.classes.param_classes as pc

import mecha.mechanisms.seasons as seasons

sir = Mechanism(name = 'sir', n = 100000)
current_mechanism = sir

""" Here we'll implement a simple sir compartmental model.  We'll have one
column called 'health' that contains 0,1 or 2 depending on whether a person
is susceptible, infected or recovered.  The infection rate, beta, will be 
proportional to the probability of being infected because the more infected
people there are, the more likely is a susceptible person to be infected.  The
recovery rate, gamma, will be a constant probability."""


beta_for_season = {'fallow' : .15,'ssc' : .15,'hiv' : .3, 'ssf' : .3}

beta = Param(
    init_now=False,
    name='beta', 
    mechanism = sir, 
    init_val = lambda self: beta_for_season[Registry.get('seasons','season').val],
    update_val = lambda self: beta_for_season[Registry.get('seasons','season').val]
    )


Probs(name='gamma', mechanism = sir, init_val = [.1,.9])


# if you don't want to use the mechanism prefix everywhere, bind the param to a variable:
health = Column(name = 'health', 
                mechanism = sir, 
                init_val = 0)


S = Cohort(
    name = 'susceptible', 
    mechanism = sir, 
    init_val = lambda self: health.eq(0), 
    update_val = lambda self: health.eq(0))

I = Cohort(
    name = 'infected', 
    mechanism = sir, 
    init_val = lambda self: health.eq(1), 
    update_val = lambda self: health.eq(1))

R = Cohort(
    name = 'recovered', 
    mechanism = sir, 
    init_val = lambda self: health.eq(2), 
    update_val = lambda self: health.eq(2))


def p_transmission ():
    n = sir.n
    
    # The probability of being an infectious agent is:
    p_infectious = sir.infected.size / n

    # Similarly, the probability of being a Susceptible agent is:
    p_susceptible = sir.susceptible.size / n

    # The probability that one meeting between two agents has one
    # infectious and one Susceptible is:
    p_one_potential_transmission = p_infectious * p_susceptible

    # Potential transmissions become actual transmissions with
    # probability beta:
    p_one_transmission = p_one_potential_transmission * sir.beta.val

    # return both probability of transmission and its complement
    return [p_one_transmission, 1 - p_one_transmission]
    

s2i = Rule(
    init_now=False,
    name = 's2i',
    mechanism = sir,
    cohort = S,
    actionlists = [ [(health,1)],[] ],
    probs = Probs(
        init_now = False,
        init_val = lambda self: p_transmission(), 
        update_val = lambda self: p_transmission() 
        ))


i2r = Rule(
    name = 'i2r',
    mechanism = sir,
    cohort = I,
    actionlists = [ [(health,2)],[] ],
    # 10% of infected recover on each time step
    probs = sir.gamma
    )


Probs(name='p_reinfection', 
      mechanism = sir, 
      init_val = [.05,.95])

r2s = Rule(
    name = 'r2s',
    mechanism = sir,
    cohort = R,
    actionlists = [ [(health,0)],[] ],
    # 10% of infected recover on each time step
    probs = sir.p_reinfection
    )
# 



# # Set initial conditions: Some of the population must be infected otherwise none can be infected!
# # Say the first 20 members of the population are infected.  Then update the cohorts.

# health.val[:50] = 1
# for cohort in [S,I,R]: cohort.update()
# print(f"Initial cohorts and sizes\nsusceptible: {S.size}, infected: {I.size}, recovered: {R.size}\n")

# print("susceptible  infected  recovered")
# # run 30 timesteps
# for i in range(30):
#     s2i.run_rule()
#     i2r.run_rule()
    
#     # recalculate the probability of transmission, which will have changed
#     # because more people have become infected
#     s2i.probs.update()  
#     # recalculate the multinomial probabilities, which depend on p_transmission
#     s2i.multinomial.update() 
    
#     # update the cohorts
#     for cohort in [S,I,R]: cohort.update()
#     print(f"{S.size:8} {I.size:10} {R.size:10}")
    



