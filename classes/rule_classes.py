#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:37:08 2022

@author: prcohen
"""

import sys,time,inspect,copy
from functools import reduce
from types import MethodType

import numpy as np
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling

import os
os.chdir("/Users/zuck016/Projects/Proposals/ASKEM")
from mecha.classes.param_classes import Param, Probs, Array, Simplex, Cohort, Gensym
from mecha.classes.utils import plot_ternary

import mecha.classes.param_classes as pc

class Rule ():
    ''' A Rule has one condition, one cohort, and one or more action lists. For 
    example, a rule might say that in the summer (the condition) all the kids 
    (the cohort) switch to t-shirts and switch to sandals (an action list of 
    two actions). 
    
    The default value of `condition` is True, meaning the rule always applies. 
    Otherwise `condition` should be callable that returns True or False. Set 
    `condition` to False to disable the rule (e.g., in ablation experiments).
    
    `cohort` must be a Cohort instance or a callable that can serve as the init_val 
    of a Cohort instance. The n of the Cohort must match the n of the rule.
    
    Actions change values in target Columns via the `assign` method of the 
    Column class. 
    
    -- True: actions apply to all rows of target columns.  
    -- A boolean column: actions apply to rows of this column that contain True
    -- A callable that returns a boolean column, used when cohort membership 
       changes dynamically
    
    If a rule contains more than one action list, it means that action list A_i 
    applies to a randomly selected proportion p_i of the cohort.  For example, 
    if there were three styles of summer garb, and `probs` contains the multinomial 
    probabilities [.2,.3,.5], then a random 20% of the cohort would wear the first 
    style, 30% would wear the second, and the remaining 50% would wear the third.  
    
    If there are k action lists, then the  multinomial must have k categories. 
    Sometimes we want to say action list A1 applies to a fraction p of the 
    cohort and nothing should be done to the remaining fraction.  This is indicated 
    by specifying `actionlists` = [A, []] with `probs` = [p, 1-p].  
    
    Actions are tuples (col,val). Executing an action is equivalent to col.asssign(val).
    
    '''
    
    def __init__(self, *args, name= None, n = None, mechanism = None, condition = True, 
                 cohort = None, actionlists = None, probs = None, update_params = False, init_now = True, **kwds):
        self.type = 'Rule'
        # anomymous rules need names too!
        self.name = name if name is not None else Gensym.make('anon')
        self.update_params = update_params
        self.args = args
        self.kwds = kwds
        self.dynamic = True # all rules are by definition dynamic
        self.mechanism = mechanism if mechanism is not None else pc.Param.guess_mechanism()
        
        # specified n overrides n inferred from mechanism
        if n is not None:
            self.n = n
        elif self.mechanism is not None:
            self.n = self.mechanism.n
        else:
            raise ValueError("If no mechanism is specified then n must be specified")
        
        self.code4callables = {'condition':inspect.getsource(condition)} if callable(condition) else {}
        
        # make the condition into an instance method if it's callable and a lambda if not
        self.condition =  MethodType(condition,self) if callable(condition) else condition
        
        # cohort must be a Cohort instance or None or True
        if isinstance(cohort,pc.Cohort):
            self.cohort = cohort
        else:
            if self.n is None:
                raise ValueError("If a Cohort is not specified, then n must be specified")
            else:
                if cohort is None or cohort is True:
                    self.cohort = Cohort(n=self.n,init_val=True)
                else: raise ValueError("The cohort argument for a rule must be a Cohort or True or its default value of None")
        
        if actionlists is None:
            raise ValueError ("A rule must have at least one actionlist")
        # if there's only one actionlist, wrap it in a list for later processing
        self.actionlists = [actionlists] if type(actionlists[0]) == tuple else actionlists 
        
        self.probs=probs
        self.multinomial = None # overridden in initialize_val if multiple action lists
        
        # register self with mechanism if mechanism is known
        if self.mechanism is not None: self.mechanism.register(self)
        
        if init_now: self.initialize_val(*args,**kwds)
             
        
    def initialize_val (self):
        
        ''' If a rule has multiple actionlists then probs must be specified to 
        select among them: One actionlist and no probabilities, or k actionlists 
        and k probabilities. Anything else throws an error.
        
        Because initializing a rule can create a Multinomial object which itself
        depends on a Probs object, repeatedly initializing a rule will created 
        multiples of these objects.  Ordinarily one wouldn't repeat intialization
        but it can happen during development or in Jupyter Notebooks. So this checks
        whether self already has a Multinomial and whether it has a Probs. If so,
        it reinitializes them rather than creating new ones.  
        '''
        
        if self.multinomial is not None:
            self.multinomial.probs.initialize_val()
            self.multinomial.initialize_val
            return
        
        if len(self.actionlists) > 1:
            if self.probs is None:
                raise ValueError("If actionlists contains more than one list, then probs must be specified.")
            elif len(self.probs.val) != len(self.actionlists):
                raise ValueError("probs must have as many probabilities as there are action lists")
            elif not isinstance(self.probs,pc.Probs):
                raise TypeError("probs must be a Probs object")
            else:
                # set dynamic to True so the multinomial will be updated
                if self.probs._val is None: self.probs.initialize_val()
                self.multinomial = pc.Multinomial(n=self.n, probs=self.probs, dynamic = True)
                self.multinomial.initialize_val()
            
        self.initargs = {'condition': self.condition, 'cohort': self.cohort, 'actionlists': self.actionlists,
                         'probs': self.probs, 'multinomial': self.multinomial, 'n':self.n, 'kwds': self.kwds}
        self.initialized = True
        
        
    def update (self,*args,**kwds):
        ''' The vals of Columns and Cohorts and the underlying probablities of
        multinomials are generally dynamic, so need updating.  But rules are
        usually called by simulators that may want to control which parameters
        are updated.  run_rule will not update its parameters unless
        update_params is True.  The default value assumes that simulators or
        another external process is responsible for updating. '''
        
        if self.update_params:
            self.cohort.rebuild(*args,**kwds)
            if self.multinomial is not None:
                self.multinomial.update(*args,**kwds)
    
    
    def run_rule (self,*args,**kwds):
        ''' Running a rule means applying actionlists to selected rows of
        a cohort.  If there is only one actionlist, then it is applied to
        all True values of the cohort.  If there are multiple action lists
        then each the ith actionlist applies to the logical_and of the cohort and
        rows that hold the ith class label in a multinomial distribution. 
        '''
        condition_met = self.condition(*args,**kwds) if callable(self.condition) else self.condition
        if condition_met:   
            
            if self.probs is None:
                for al in self.actionlists:
                    for col,val in al:
                        col.assign(val,selected = self.cohort.val)
            else:
                for i in range(len(self.actionlists)):
                    al = self.actionlists[i]
                    if al != []:
                        for col,val in al:
                            col.assign(val,selected = np.logical_and(self.cohort.val,self.multinomial.eq(i)))
                            
    def describe (self):
        if self.name is not None: print(f"name: {self.name}")
        if self.mechanism is not None: print(f"mechanism: {self.mechanism.name}")
        print(f"condition: {self.condition}")
        cohort = self.cohort.name if self.cohort is not None else None
        print(f"cohort: {cohort}")
        print(f"probs:{self.probs}")
        
        
                  




class Influence (Rule):
    ''' An Influence is a Rule that updates a Simplex.  Its name connotes the idea
    of affecting beliefs or subjective probabilities.  Influences have all the 
    methods of Rules, but the `run` method works differently because the structure
    it operates on is a Simplex, not a Column.  
    
    Like a Rule, an Influence has an optional condition that must be True for any
    action to be taken.  And Influences operate on Cohort members, not on non-members.
    Influences also can have probabilistic effects, but this exposes an important 
    difference between Rules and Influences:  A Rule might stipulate that 1/3rd of
    the Cohort, selected at random, is affected by each of actions A,B and C.  The
    underlying multinomial is updated on each step and a random selection is made 
    again so that a cohort member isn't "stuck" with the same action on every step.
    However, Influences have a kind of persistence that would be defeated by selecting
    a new action at each time step.  For example, if a farmer is influenced to plant
    tomatoes on one time step, then we want the influence to continue on the next
    time step.  There might be *competing* influences, but any given influence should
    "say the same thing" on each time step. To ensure this, the multinomial is *not*
    updated on each time step. 
    
    Another difference between Rules and Influences is the form of actions.  The
    actionlists of Rules identify a Column and the value to which selected rows 
    should be set.  Influences instead return destinations toward which selected
    points in the Simplex are moved.  Optionally, Influences may return a vector
    of rates (see Simplex documentation).  The form of an actionlist is therefore
    [ (s,d,r) ] where s is a simplex, d is a single destination or a 2D array of
    destinations, and an optional rate -- a single scalar or a vector of rates.  
     
    '''
        
    def __init__(self, *args, simplex = None, **kwds):
        self.simplex = simplex
        super().__init__(**kwds)
        
        if self.probs:
            self.multinomial.dynamic = False
    
        
    def update (self,*args,**kwds):
        ''' The vals of Columns and Cohorts and the underlying probablities of
        multinomials are generally dynamic, so need updating.  But rules are
        usually called by simulators that may want to control which parameters
        are updated.  run_rule will not update its parameters unless
        update_params is True.  The default value assumes that simulators or
        another external process is responsible for updating. '''
        
        if self.update_params: self.cohort.rebuild(*args,**kwds)
            
    
    def run_rule (self,*args,**kwds):
        ''' Running a rule means applying actionlists to selected rows of
        a cohort.  If there is only one actionlist, then it is applied to
        all True values of the cohort.  If there are multiple action lists
        then each the ith actionlist applies to the logical_and of the cohort and
        rows that hold the ith class label in a multinomial distribution. 
        '''
        
        condition_met = self.condition(*args,**kwds) if callable(self.condition) else self.condition
        if condition_met:   
            
            if self.probs is None:
                for al in self.actionlists:
                    for simplex,destination,rate in al:
                        simplex.update(destination,rate,selected = self.cohort.val)
            else:
                for i in range(len(self.actionlists)):
                    al = self.actionlists[i]
                    if al != []:
                        #print(f"{i}. multinomial: {self.multinomial.val}\ncohort:         {self.cohort.val.astype(int)}\nselected:       {np.logical_and(self.cohort.val,self.multinomial.eq(i)).astype(int)}")
                        for simplex,destination,rate in al:
                            #print(f"destination: {destination}\n")
                            simplex.update(destination,rate,selected = np.logical_and(self.cohort.val,self.multinomial.eq(i)))















