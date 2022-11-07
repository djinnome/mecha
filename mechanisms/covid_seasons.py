#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 14:32:55 2022

@author: prcohen
"""

import os, time
os.chdir("/Users/zuck016/Projects/Proposals/ASKEM")
import mecha

import pandas as pd
import numpy as np

from mecha.classes.registry import Registry
from mecha.classes.param_classes import Param
from mecha.classes.rule_classes import Rule
from mecha.classes.mechanism_classes import MechMap, Sim, Mechanism
from mecha.classes.utils import plot_series, plot_multiseries,plot_replicates

import mecha.mechanisms.seasons as seas
import mecha.mechanisms.covid as covid


sir = covid.sir
seasons = seas.seasons

# You can initialize this way:
seasons.initialize()
sir.initialize()

# Or this way:
#Registry.initialize_all_mechanisms()


d,g = MechMap.make(sir,seasons,viz=True)
display(g)

def setup_sir ():
    sir.health.val[0:50] = 1
    
def setup_seas ():
    seasons.start_label = seas.starting_season()
    
def probe_fn ():
    return [sir.s2i.multinomial.probs.val,sir.infected.size,seasons.season.curr,sir.beta.val]

probe_labels = ['p','sir.infected.size','seasons.season','sir.beta']
                

sim = Sim(
    n_trials = 3,
    n_steps = 700,
    setup = [(sir, setup_sir),(seasons, setup_seas)],
    probe_fn = probe_fn,
    probe_labels = probe_labels
    )


sim.run()


df = pd.DataFrame(sim.record,columns=['trial','step']+probe_labels)
df1 = df[['sir.beta','step','trial','sir.infected.size']]
plot_replicates(df1,x='step',standardize=True,xlabel=f"days elapsed since day {seasons.clock.initargs.get('init_val')}",ylabel='standardized (Z)')
#plot_multiseries(df1,x='step',standardize=True,xlabel='days since start of sim',ylabel='standardized (Z)')
