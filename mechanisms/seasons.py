#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 15:24:08 2022

@author: prcohen
"""

import os
os.chdir("/Users/zuck016/Projects/Proposals/ASKEM")

import numpy as np
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling

from mecha.classes.registry import Registry
from mecha.classes.param_classes import Param,  Probs, Counter, Switch, Cycle
from mecha.classes.mechanism_classes import Mechanism, Sim
from mecha.classes.rule_classes import Rule

import mecha.classes.rule_classes as rule_classes
import mecha.classes.param_classes as pc
        
seasons = Mechanism('seasons')
current_mechanism = seasons

Cycle(
      name = 'clock',
      mechanism = seasons,
      labels = np.arange(1,366),
      init_val = 40,
      update_val = True,
      doc = '''This cycles through the seasons.'''
    )



Param(
    name = 'dagana_season_transition_days',
    mechanism = seasons,
    init_val = [32, 182, 288, 335],
    doc = '''This stores the days on which agricultural seasons change.  
    Different parts of the country will have different transition days.'''
    )
    

def starting_season ():
    '''Returns the starting state of the season Cycle based on the clock value.'''
    c = seasons.clock.val
    d = seasons.dagana_season_transition_days.val
    if c < d[0] or c >= d[3]: return 'fallow'
    if c < d[1]: return 'ssc'
    if c < d[2]: return 'hiv'
    if c < d[3]: return 'ssf'
    
        
season = Cycle(
      name = 'season',
      mechanism = seasons,
      labels = ['fallow','ssc','hiv','ssf'],
      init_val = lambda self: starting_season(),
      update_when = lambda self: seasons.clock.curr in seasons.dagana_season_transition_days.val,
      doc = '''This cycles through the seasons'''
    )


# test
# def test():
#     for i in range(10):
#         seasons.clock.update()
#         seasons.season.update()
#         covid.beta.update()
#         print(f"day:   {seasons.clock.val} \t\t season: {seasons.season.curr} \t\t beta: {covid.beta.val}")


# def probe_fn ():
#     print(seasons.clock.val,seasons.season.curr_label)
#     return [seasons.clock.val,seasons.season.curr_label]

# sim = Sim(
#     n_trials = 2,
#     n_steps = 100,
#     setup = [(seasons,None)],
#     probe_fn = probe_fn,
#     probe_labels = None
#     )

# sim.run_sim()







