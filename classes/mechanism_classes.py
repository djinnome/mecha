#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:16:54 2022

@author: prcohen
"""

import sys,time,inspect,copy


from types import SimpleNamespace
from functools import reduce
import collections


import numpy as np
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling


# It's necessary to import an existing Registry rather than create it here, otherwise
# every module that calls this one will create a new registry, so different mechanism
# modules wouldn't have their objects combined in one registry.

from mecha.classes.registry import Registry
import mecha.classes.param_classes as pc
import mecha.classes.rule_classes as rule_classes

# To show graphs of associations between parameters we need graphviz, but comment
# this out if you don't want to show graphs or don't want the dependency on grapviz
import graphviz

# TODO Explore how to import previously defined instances into a mechanism.  The
# importing itself is easy but suppose object X was defined in mechanism A and
# imported into mechanism B: Which mechanism is responsible for updating X in
# simulations?  I guess A should do it???



   
class Mechanism (SimpleNamespace):
    ''' A mechanism is a 'self-contained' part of a simulation.  The idea is to 
    break a complicated Big Mechanism such as an agricultural value chain into
    manageable units that can be worked on nearly independently.  Each mechanism
    is a namespace so its parameters and rules can be referred to with dot 
    notation (e.g., covid.beta where covid is a mechanism and beta is a param).
    Mechanisms can import parameters from other mechanisms. 
    
    Each mechanism has a parameter, n, that specifies the length of any columns
    it deals with, e.g., the size of a population on which the mechanism operates.
    '''
    def __init__(self, name, n = None):
        self.name=name
        self.n = n
        self.type = 'Mechanism'
        Registry.register(self)
        # self.objects keeps track of the objects (Params, Rules, etc.) in self
        self.objects = collections.OrderedDict()
        self.rules = []
        
    def register (self,obj):
        ''' Objects (e.g., Params) are registered with their mechanisms.  '''
        
        name = obj.name
        # Check whether param already exists
        if self.objects.get(name) is not None:
            print(f"WARNING:  Overwriting parameter {name}")
        self.objects.update({name:obj})
        if isinstance(obj,rule_classes.Rule): 
            self.rules.append(obj)
        self.__dict__.update({name:obj})
        
       # print (f"\nRegistry:\n{[m.objects.keys() for m in Registry.mechanisms.values()]}")
    
    def update (self,*args, **kwds):
        ''' Updates the values of all the dynamic objects in the mechanism.'''
        for obj in self.objects.values():
            if obj.dynamic: 
                obj.update(*args,**kwds)
    
    def run_rules (self,*args, shuffle_rules=False, **kwds):
        # shuffle the rules if required and then run each rule
        if shuffle_rules: rng.shuffle(self.rules)
        for rule in self.rules: rule.run_rule()
        
    def initialize (self):
        ''' When there are circular dependencies between mechanisms, all the 
        mechanisms are read in before any mechanism is initialized. This intializes
        all the objects in a mechanism. Some params are created when others are
        initialized. These need to know `current_mechanism` so they will register
        with this mechanism.'''
        current_mechanism = self
        for obj in self.objects.copy().values():
            obj.initialize_val()
     
    def reset(self,*args,**kwds): 
        ''' Resets the original values of all variables. To re-evaluate callables use rebuild.'''
        for name,obj in self.objects.items():
            if obj.type != 'Rule':
                obj.reset(*args,**kwds)
       
    def rebuild (self,*args,**kwds):
        ''' This rebuilds all the params in the Mechanism. Because self.objects
        is an ordered dict, this rebuilds the objects in the same order that they
        were created.'''
        for name,obj in self.objects.items():
            if obj.type != 'Rule':
                obj.rebuild(*args,*kwds)
                
    def describe (self):
        print(f"Mechanism {self.name}\nn = {self.n}\nobjects =")
        for k,v in self.objects.items():
            print(f"  {k}:  {v}")
                
                
                

class Sim ():
    ''' A trial involves 'running' one or more Mechanisms for n_steps timesteps. 
    Within each timestep, all the mechanism parameters are updated, all the rules
    are executed, and data are gathered.  If updates require keyword arguments, 
    pass them through Sim **kwds. The Sim parameter `setup` is a list of 
    tuples in which the first element is a mechanism and the second is a function 
    that sets up the initial conditions for that mechanism before a trial begins.
    
    The order in which mechanisms run can be randomized at the beginning of a 
    trial.  (The order could also be randomized on each time step but this seems
    excessive and is not allowed, currently.)  The order of the application of
    rules within a mechanism can also be randomized and this is done on each
    timestep, if shuffle_rules is True.
    '''
    def __init__(self, n_trials, n_steps, setup, probe_fn = None, probe_labels = None, 
                 shuffle_rules = False, shuffle_mechanisms = False, **kwds):
        self.n_trials = n_trials
        self.n_steps = n_steps
        self.setup = setup
        self.probe_fn = probe_fn
        self.probe_labels = probe_labels
        self.shuffle_rules = shuffle_rules
        self.shuffle_mechanisms = shuffle_mechanisms
        self.kwds = kwds
        
        self.trial = -1
        self.step = -1
        
        self.record = []
        
    def run_trial (self):
        ''' '''
        for step in range(self.n_steps):
            self.step += 1
            
            for mechanism, _ in self.setup:
                
                # update the values of all params etc.
                mechanism.run_rules ()
                mechanism.update ()
                
                
            
            # build a row of data and add it to self.record
            if self.probe_fn:
                row = [self.trial,self.step]
                row.extend(self.probe_fn())
                self.record.append(row)
            
    def run (self):
        for trial in range(self.n_trials):
            self.trial += 1
            self.step = -1
            
            # the order in which mechanisms run can be shuffled initially
            if self.shuffle_mechanisms :  rng.shuffle(self.setup)
            
            # rebuild the parameter values for each parameter then set up 
            # initial conditions for each mechanism to run
            for mechanism, init_conds in self.setup: 
                mechanism.rebuild()
                if init_conds is not None: init_conds(**self.kwds)
                
        
            self.run_trial()  
    
    
    

'''Large models might incorporate many nechanisms, each with its own namespace, 
    parameters and rules.  Keeping track of all these objects can be taxing, espcially
    when large models are developed by many people simultaneously.'''
    
    
def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])



class MechMap():
    '''
    The MechMap class contains functions to find dependencies between parameters.  
    It performs a simple static analysis to find named and anonymous parameters  
    and produces a two-level dict {p1 : {r1_1 : p1_1,...}, p2: {...} ...} in 
    which p1 and p1_1 are parameter names, and r1_1 is a directional relation 
    between them. Because lambda expressions are common in the Mechanisms package,
    it unpacks callables and searches the resulting strings for names of parameters.
    It cannot find a Param that is not registered with a Mechanism (see the
    guess.mechanism method of Param.) '''
    
    @classmethod
    def make (MechMap, *mechanisms, viz = False):
        ''' This accepts any number of mechanisms, merges their objects and finds
        dependencies between objects. This prefixes each object name with its 
        mechanism name.
        '''
        def merge_dicts (*dicts):
            return {k: v for d in dicts for k, v in d.items()}

        names_without_prefixes = [y for x in [m.objects for m in mechanisms] for y in x]
        merged_objects = merge_dicts(*[{m.name+'.'+k : v for k,v in m.objects.items()} for m in mechanisms])
        prefix2name = merge_dicts(*[{k: m.name+'.'+k for k in m.objects.keys()} for m in mechanisms])
        
        dependencies = MechMap.get_dependencies(merged_objects, names_without_prefixes)
        
        if viz:
            graphs = MechMap.make_gvgraph(dependencies,name_dict=prefix2name)
            return dependencies,graphs
        else:
            return dependencies,None
        
    
    @classmethod 
    def get_mech_dependencies (MechMap,mechanism):
        return MechMap.get_dependencies(mechanism.objects, mechanism.objects.keys())
    
    @classmethod
    def get_dependencies (MechMap, objects, names):
        ''' `objects` is a list of mechanism objects (e.g., params, rules,...). `names`
        is typically a list of names of objects.  It is needed because get_dependencies
        searches strings returned by inspect.getsource (i.e., string representations of
        source code) to find dependencies between objects.  This returns a dict of the 
        form {head: {rel: tail}} where head is an object name, tail is also a name or 
        a list of names, and rel is a relation between them.  The relation is one of 
        those in `to_expand`.'''
                                                                     
        to_expand = ['actionlists','cohort','condition','conditions','init_val',
                     'multinomial','probs','update_val','update_when','simplex']
        to_explore = list(objects.values())
        explored= []
        associations = {}
        
        def associations_for (obj,names):
            slots = {k:obj.__dict__.get(k) for k in to_expand if obj.__dict__.get(k) is not None} 
            
            if slots.get('actionlists'):
                a = flatten(slots['actionlists'])[0]            
                for item in a:
                    if isinstance(item,pc.Param) and item not in to_explore and item not in explored: 
                        to_explore.append(item)
                slots['actionlists'] = [item.name for item in a if isinstance(item,pc.Param)]
            
            for k,v in slots.items():
                if callable(v):           
                    source = obj.code4callables.get(k)
                    slots[k] = [name for name in names if (name in source and name != obj.name)]
                elif isinstance(v,pc.Param):
                    if v not in to_explore and v not in explored: 
                        to_explore.append(v)
                    slots[k] = v.name
            final = {k:v for k,v in slots.items() if v != []}     
            return final
        
        while to_explore != []:
            obj = to_explore.pop()
            explored.append(obj)
            associations[obj.name] = associations_for(obj,names)
        
        return associations
    
    @classmethod
    def make_gvgraph (MechMap, dependencies, rel_dict = None, name_dict = None, size = '10'):
        ''' dependencies is a dict of the form {head: {relation: X}} where
        X can be a string or a list of strings. rel_dict maps the rels in
        dependencies to other strings that might be more descriptive. name_dict
        similarly maps names to other names, particularly to add a mechanism prefix
        to node labels. '10' is a good default size for jupyter notebooks'''
       
        gv = graphviz.Digraph(
            graph_attr = {'size' : size},
            node_attr={'shape': 'box','margin' : '.2'})
        
        for head,rels in dependencies.items():
            _head = head if name_dict is None else (name_dict.get(head) or head)
            gv.node(_head,label=_head)
            if rels != {}:
                for rel,tail in rels.items():
                    _rel = rel if rel_dict is None else (rel_dict[0].get(rel) or rel)
                    if type(tail) == str:
                        _tail = tail if name_dict is None else (name_dict.get(tail) or tail)
                        gv.edge(_tail,_head,_rel)
                    elif type(tail) == list:
                        if all([type(tl) == str for tl in tail]):
                            for tl in tail: 
                                _tl = tl if name_dict is None else (name_dict.get(tl) or tl)
                                gv.edge(_tl,_head,_rel)
                        else:
                            gv.edge(str(tail),_head,_rel)
        return gv
        




