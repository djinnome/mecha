#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:31:00 2022

@author: prcohen
"""

import sys,time,inspect,copy


from types import SimpleNamespace, MethodType

import numpy as np
from numpy.random import default_rng
rng = default_rng(1108) # stuff for random sampling
from functools import partial

import os
os.chdir("/Users/zuck016/Projects/Proposals/ASKEM")
from mecha.classes.registry import Registry
    


#TODO: rework the following
'''
This module contains several classes of parameter.  All parameters inherit from
the class Param.  A top-level distinction is made between StaticParam and 
DynamicParam, of which only the latter takes an update function.  The subclasses
of StaticParam and DynamicParam are 

-- StaticProbs and DynamicProbs, which hold an array of probabilities that sum to 1.0;
-- Multinomial, which generates and maintains a multinomial distribution
-- Switch, which can represent a Finite State Machine with conditional transitions
-- Cycle, which proceeds through a list of states such as days of the week, returning to the first
-- Counter, which starts at a specified nunber and increases it by a specified increment

All Params take an init_val argument that initializes the value of the param.
Additionally, dynamic Params take an update_val argument.  Both init_val and
update_val can be callable.
    
'''

#=============================================================================
# Param
#=============================================================================

# TODO: should more methods be class methods not instance methods

# TODO: Why can't we write docstrings for class instances? I hacked it with
# making doc a parameter and binding it to __doc__.  I also made help a 
# property but surely there's a more standard way?


class Gensym ():
    ''' Returns the "next" integer with an optional prefix.'''
    curr = 100 
    @classmethod
    def make (Gensym, prefix = 'anon'):
        Gensym.curr += 1
        return prefix+'_'+str(Gensym.curr)
    
    
class Param ():
    """
    Params hold and update values. At present, these values must be integers,
    float, booleans or np.arrays.  The values of parameters can be changed in 
    three ways:
        
        -- By assignment
        -- By running `reset` or `rebuild`, in which case `init_val` is re-evaluated
        -- by running `update`, in which case `update_val` is evaluated
        
    Thus, the initial value of a param can be set by a different mechanism 
    than subsequent values; e.g., initial value might be a constant, `update_val`
    might be a callable that depends on dynamic conditions.
        
    If `update_val` is None, then calling `update` throws an error.  A param is
    classified as dynamic iff update_val or update_when is callable, or if it
    is "inherently" dynamic (e.g., Counters).
    
    Updating happens only when `update_when` is True. This is the default 
    value of `update_when`. One way to disable updating a Param (e.g., in 
    an ablation experiment) is to set `update_when` to False. 
    
    `update_when` also can be callable, so dynamic conditions can dictate when to update.  
    
    
    All callables are turned into instance methods so they can access self.  
    But this requires its first argument of callables to be something that binds 
    to self.  Three legal configurations are:
        
        1) Param(init_val = lambda cls, *args, **kwds: ....)   
        
        2)  def foo (): ...
            Param(init_val = lambda cls: foo() ...)
                  
        3)  def foo (cls): ...
            Param(init_val = foo, ...)
            
    
    By default, params are initialized when they are created.  However, this can
    cause problems when there are circular dependencies among params in different
    modules.  In such cases, set init_now = False and call initialize_val after
    all dependencies are resolved (e.g., after all modules are loaded), or rely 
    on the intialize methods of Mechanism or Registry. 
    
    Param values are accessed several ways:
        -- self.val, self.curr, self.get() and self._val all return the current value
        -- self.prev and self._previous return the previous value
        
    Some parameters (e.g., Switch, Cycle) may have labels (e.g., 'spring', 'summer',...)
    associated with values, in which case:
        
        -- self.curr_label returns the current label
        -- self.prev_label returns the previous label
        
    For Cycle parameters, where known values succeed each other in a fixed order,
    self.next and self.next_label return the value/label that will be current on 
    the next update.
    
    curr and prev have two interpretations.  The one that's implemented is 
    illustrated by prev_1:  prev holds the last value that curr held before it 
    changed. An alternative interpretation -- not implemented but easily done 
    if there's a need --is illustrated by prev_2, which holds the value curr held 
    at the last time step. 
    
    curr     0 1 0 0 
    prev_1   0 0 1 1
    prev_2   0 0 1 0
    
    """         
    
    def __init__(self, *args, name=None, mechanism=None, init_val = None, update_val = None, 
                 update_when = True, dynamic = False, doc = None, init_now = True, **kwds):
        self._val = None
        self.initialized = False
        self.type = 'Param'
        self.doc = doc
        self.args=args
        self.kwds = kwds
        
        # anomymous objects need names too!
        self.name = name if name is not None else Gensym.make('anon')
        
        # if mechanism is None, search for current_mechanism in the stack
        self.mechanism = mechanism if mechanism is not None else Param.guess_mechanism()
        # register self with mechanism 
        if self.mechanism is not None: self.mechanism.register(self)
        
        # The MechMap class builds a map of dependencies between parameters.  
        # Some of these dependencies are hidden in the code of callables. 
        # Here we inspect and save the code in a dict:
            
        self.code4callables = {k:inspect.getsource(v) 
                               for k,v in zip(['init_val','update_val','update_when'],
                                              [init_val,update_val,update_when])
                               if callable(v)}
        
        self.init_val = MethodType(init_val,self) if callable(init_val) else init_val
        self.update_val = MethodType(update_val,self) if callable(update_val) else update_val
        self.update_when = MethodType(update_when,self) if callable(update_when) else update_when
        
        self.initargs = {}
    
        # A parameter is dynamic if dynamic = True or if update_val or update_when is callable,
        # or when the parameter is a Counter, Cycle or Switch (see their class definitions)
        self.dynamic = dynamic is True or callable(self.update_val) or callable(update_when)
        
        if init_now: self.initialize_val(*args,**kwds)
    
    def initialize_val (self,*args, **kwds):
        self._val = self.init_val(*args,**kwds) if callable(self.init_val) else self.init_val
        self._previous = None
        self.make_initargs()
        self.initialized = True
        
        
    def make_initargs (self, **kwds):            
        # By storing self.initargs we can reset the Param. 
        self.initargs.update({
            'init_val': self.init_val, '_val' : copy.deepcopy(self._val), 
            '_previous' : copy.copy(self._previous), 'update_val': self.update_val, 
            'update_when': self.update_when, 'args': self.args, 'kwds' : self.kwds})        
        if kwds is not None:
            self.initargs.update(kwds)
    
    @classmethod
    def guess_mechanism (self):
        ''' If self.mechanism is None, see whether  `current_mechanism` is in 
        the calling frame and if so, use it.  This is especially useful to ensure 
        that anonymous params are registered with whichever is the `current_mechanism`.'''  
        current_mechanism = None
        for f in inspect.stack():
            previous_frame_locals = f[0].f_locals
            current_mechanism = previous_frame_locals.get('current_mechanism')
            if current_mechanism is not None:
                break
        if current_mechanism is None: 
            pass
            #print("WARNING: No `current_mechanism` was found. Parameters may not be registered with a mechanism.")
        return current_mechanism
    
    
    # the following "lazy" getter makes it possible to query val without 
    # having to manually initialize the param, which can be inconvenient during
    # development.  When there are circular dependencies between modules this will
    # sometimes raise an error
    
    @property
    def lval(self):
        if self._val is None:
            print(f"lazy evaluation of {self.name} _val")
            self.initialize_val(*self.args,**self.kwds)
        return self._val
    
    @property
    def val(self):
        return self._val
    
    @val.setter
    def val(self, value):
        self._previous = self._val
        self._val = value

    @val.deleter
    def val(self):
        del self._val
        
    @property
    def curr(self):
        return self.val
    
    @property
    def prev(self):
        return self._previous
        
    
    def update (self,*args,**kwds):
        ''' Throws an error if self.update_val is None. update is intended only for 
        updating, not resetting or rebuilding the parameter val.  E.g., if init_val
        is callable and update_val is None, then "updating" probably means re-evaluating
        init_val.
        '''
        if self.update_val is None:
            raise ValueError(f"In param {self.name}. update called but update_val is None. Try reset or rebuild instead.")
            
        update_now = self.update_when(*args,**kwds) if callable(self.update_when) else self.update_when
        if update_now:
            new_val = self.update_val(*args,**kwds) if callable(self.update_val) else self.update_val
            self._previous = self._val
            self._val = new_val   
    
    def reset(self): 
        ''' Resetting a Param affects only its _val attribute.  It is reset
        to the value it had when the Param was first created. If init_val is
        callable, consider `rebuild`, which re-evaluates the callable.
        
        Because of lazy evaluation of _val, calling reset before _val has a value
        will have no effect.  
        '''
        self._val = copy.copy(self.initargs.get('_val'))
        self._previous = copy.copy(self.initargs.get('_previous'))
        
        
    def rebuild (self,update_initargs = False):
        ''' Rebuilding a parameter replaces some attributes of the Param with
        their original values (i.e., the values they had when the Param instance
        was created. These attributes are init_val, update_val and update_when.
        However, rebuild *changes* only one thing, namely, the _val attribute.
        If init_val is callable, then rebuild will re-evaluate init_val. (If init_val 
        is a value rather than a callable then rebuild and reset are equivalent.) 
        
        If the callable has a random element (e.g., init_val = lambda: rng.random()) 
        then rebuild generally will not produce the same _val as the original.  
        (Setting the numpy rng random seed can ensure that random functions produce 
        replicable results.)  If the callable wraps another parameter whose value 
        has changed, then, again, rebuilding will not recreate the original _val. 
        In this case, consider Mechanism.rebuild, which rebuilds every parameter 
        in the order the parameters were created. 
        
        Because of lazy evaluation of _val, calling rebuild before _val has a value
        will have strange effects.  
        
        '''
        
        if self._val is None:
            # _val has never been initialized
            self.initialize_val(*self.args,**self.kwds) 
        else:
            # replace attributes with their original values
            self.init_val = self.initargs.get('init_val')
            self.update_val = self.initargs.get('update_val')
            self.update_when = self.initargs.get('update_when')
            
            # rebuild initial value
            self.initialize_val(*self.args,**self.kwds) 
            
            # if update_initargs is True then use the new _val as the 'original' 
            if update_initargs: self.initargs['_val'] = copy.copy(self._val)
        
    def describe (self):
        print(f"\nname: {self.name}")
        if self.mechanism is not None:
            print(f"\nmechanism: {self.mechanism.name}")
        print(f"init_val: {self.init_val}\n_val: {self._val}\n_previous: {self._previous}")
        print(f"update_val: {self.update_val}")
        print(f"update_when: {self.update_when}")
        print(f"kwds: {self.kwds}")
        if self.doc is not None: print(f"doc: {self.doc}")


#=============================================================================
#  Probs
#=============================================================================

class Probs(Param):
    ''' The _val of a Probs parameter is a numpy array of elements that sum
    to 1.0.  A Probs parameter checks this property when an instance is created
    and whenever an assignment to _val is made.'''
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.type = 'Probs'
        
    def initialize_val (self,*args,**kwds):
        super().initialize_val(*args,**kwds)
        # in case the original init_val was a list instead of array:
        self.val = np.array(self.val)
        self.check(self._val)
        self.initialized = True
                   
    @property
    def val(self):
        return super().val
    
    @val.setter
    def val(self, value):
        self.check(value)
        self._previous = self._val
        self._val = value

    def check (self,value):
       """checks whether self.val is an array of numbers that sums to 1.0."""   
       v = np.array(value) 
       try:
           s = np.sum(v)
       except:
           raise ValueError(f"Cannot sum self.val : {v}")
           
       # a cheaper form of np.isclose
       if np.round(s,8) != 1:
           raise ValueError(f"{v} does not sum to 1.0")
       return v
   
    def reset(self,*args,**kwds): 
        super().reset(*args,**kwds)
        self.check(self._val)
        


#=============================================================================
# Column-like params :  Column
#=============================================================================

class Column (Param):
    """ Columns represent attributes of a population; for example, an age column
    would represent the age of each member of a population.  Columns are implemented
    as one-dimensional numpy arrays that (typically) are the same length (n) as a 
    population.  
    
    If n is not specified, this will try to infer it from its mechanism.  If no
    mechanism is specified either, or if the mechanism has no n, then an error
    is raised.
    
    
    Columns are initialized in several ways:
        
        -- given n and x which is int, float or boolean, set _val to an np.array of x
        -- given x which is a list or np.ndarray, set the _val to np.array(x)
        -- if init_val is None, then a column of zeros of length n is created
        -- if init_val is callable, _val is set to the result of evaluating init_val
        
    Columns can test their values with eq, ge, le, ne, gt, and lt; for example,
    age.gt(18) returns a boolean array in which True at an index indicates that 
    the Column value at that index is greater than 18.
    
    Columns are updated with the `update_val`and `update_when` methods combined 
    with an optional selector.  `update_when`gives the conditions for updating 
    the column, `update_val` gives new value(s) for the column, and the selector 
    says which rows of the column should receive new values.  For example, if 
    'adult' and 'age' are columns, then we might want to update adult to True 
    for rows for which age >= 18.  `update_when` would specify the conditions 
    in which this would happen (e.g., every month), `update_val` would be True and 
    `selector` would be a boolean column that tests which rows have age >= 18.
    
    It is tempting to allow `selected` to be callable so that the selected rows
    can be updated dynamically.  Currently this is not allowed.  The reason is that 
    often we want to change the same rows in several columns.  If each column
    were required to figure out which rows to change, then 1) the selection
    computation would have to be repeated in each affected column, 2) if the
    selection computation includes a random element then we cannot guarantee that
    the same rows are selected in each of several columns.  To force the column
    to compute its own `selected`, use `update_val` and `update`.
    
    In general, `selected` is itself a boolean column that can be bound to several 
    columns as a keyword argument.  Changing this `selected` column ensures that 
    all the columns to which it is bound have the same `selected` rows. 
    
    """

    def __init__(self, n = None, **kwds):
        self.n = n  # the number of rows
        super().__init__(**kwds)
        self.type = 'Column'
        
        
        ''' Elementary comparison operators for columns. Calling these before 
        _val is initialized will raise an error.
    
        -- For a Column c, c.eq(17) returns a boolean column where True means
           the row value is 17, i.e., c.eq(17) is equivalent to c == 17
    
        -- For a MultiColumn, c.eq(0,17) returns a boolean column for the 0th column
           of the multicolumn; i.e., c.eq(0,17) is equivalent to c[:,0] == 17 '''
        
        self.eq = partial (self.op, fn = (lambda x,y : x == y) )
        self.ge = partial (self.op, fn = (lambda x,y : x >= y) )
        self.le = partial (self.op, fn = (lambda x,y : x <= y) )
        self.ne = partial (self.op, fn = (lambda x,y : x != y) )
        self.gt = partial (self.op, fn = (lambda x,y : x > y) )
        self.lt = partial (self.op, fn = (lambda x,y : x < y) )
    
               
    def op (self,y,fn):
        try:
            if callable (y):
                return fn(self._val,y.__call__())
            else:
                return fn(self._val,y)
        except TypeError:
            print (f"Cannot compare Column {self.name} to {y} probably because {self.name} isn't initialized yet")
        
        
    def initialize_val (self,*args,**kwds):
        
        if self.n is None and self.mechanism is not None and self.mechanism.n is not None:
            self.n = self.mechanism.n
        
        if callable(self.init_val):
            self._val = self.init_val(*args,**kwds)
        elif type(self.init_val) in [list, np.ndarray]:
            self._val = np.array(self.init_val)
        elif type(self.init_val) in [int,float,bool]:
            if self.n is not None:
                self._val = np.full(self.n,self.init_val)
            else:
                raise ValueError("Cannot create Column because n is not given and cannot be inferred from Mechanism.")
        else:
            raise ValueError("init_val must be an np.array, an int, float or boolean, or a callable.")
        
        # even if n wasn't specified explicitly we can now get it from self._val, and
        # if it was specified explicitly, the value from self._val overrides
        if self.n is not None and self.n != len(self.val):
            print (f"WARNING: Column {self.name} has a different length than specified by n")   
        self.n = len(self._val)
        self._previous = None
        super().make_initargs(n = self.n)
        self.initialized = True
        
        
    def assign (self,val,*args,**kwds):
        self.assign_helper(val=val,col=self._val,*args,**kwds)
     
    def assign_helper (self, val, col, *args,**kwds):   
        ''' 
        Assigns val, which can be a single value or an array, to selected rows
        of self. 
        '''
        self._previous = copy.copy(self._val)
        
        selected = kwds.get('selected')
        if selected is None:
            col[:] = val
        else:
            val_type = type(val)
            if val_type in [int,bool,float]:
                col[selected] = val
            elif val_type == np.ndarray:
                col[selected] = val[selected]
            else:
                raise ValueError (
                    f"Trying to assign something other than int, bool, float, or numpy array to {self.name}")



#
#=============================================================================
# Column-like params :  Cohort
#=============================================================================

class Cohort (Column):
    ''' A Cohort is a boolean Column that is used to identify rows that satisfy
    a logical expression.  `init_val` must be True, False or a callable:
        
        -- True/False: self.val is a column of True/False
        -- callable: self.val is the result of evaluating the callable
    '''
    def __init__(self,*args,**kwds):
        super().__init__(*args,**kwds)
        self.type = 'Cohort'
    
    def initialize_val(self,*args,**kwds):
        super().initialize_val(*args,**kwds)
        self.check(self._val)
        Param.make_initargs(self,n = self.n)
        self.initialized = True
    
    @property
    def val(self):
        return super().val
    
    @val.setter
    def val(self, value):
        self.check(value)
        self._previous = self._val
        self._val = value
    
    @property
    def size (self):
        return np.sum(self._val)
    
    @property
    def members (self):
        return np.where(self._val)[0]
    
    def check (self,value):
        if self._val.dtype != bool:
            raise TypeError(f"Initializing Cohort {self.name}: init_val does not evaluate to boolean column")

        



#=============================================================================
# Column-like params : Multinomials
#=============================================================================

def make_multinomial_distribution (n, probs):
    """
    If N is the desired sample size, then a multinomial distribution M that
    respects the K probabilities p_0...p_K-1 will have p_i * N instances of i.
    So we create a numpy array that has p_i * N instances of i for each i.
    Then we shuffle the array to ensure that each element is a random draw from M.
    The entries in the array will be integers 0...K-1 that can be used as indices.
    For large N this is faster than numpy's random multinomial by a factor of ten.
    If there are just two probs, this generates a binomial, which is faster.  The
    multinomial method generates "nearly exactly" n*p_i elements i, whereas the
    binomial method is sloppier. 
    """
    m = len(probs)
    if m == 2:
        p = probs[0]
        return (rng.random(n) > p).astype(int)
    
    s = np.arange(len(probs))[::-1]
    a = np.empty((n),dtype=int)
    q=list((np.array(probs).cumsum()*n).astype(np.int32))[::-1]
    for i in range(m):
        a[:q[i]] = s[i] # fill a with the right number of each index from s
    np.random.shuffle(a)
    return a

# p = np.array([0.1, 0.1, 0.3, 0.2, 0.05, 0.05, 0.2])

# t = time.time()
# for i in range(10000):
#     a = make_multinomial_distribution (10000, p)
# print(time.time()-t)
# print(a[:30])
# u,c = np.unique(a,return_counts=True)
# print(u,c)

class Multinomial (Column):
    ''' 
    This maintains a multinomial distribution of length n given probs, a Probs 
    object.  If the underlying probs change then the distribution must be recreated. 
    No update_fun needs to be supplied: Updating means rebuilding the multinomial
    distribution if the underlying Probs have changed
    '''
    
    def __init__(self, probs = None, **kwds):
        self.type = 'Multinomial'
        self.probs = probs
        super().__init__(**kwds)
        
    def initialize_val (self,*args,**kwds):
        # try to infer n from mechanism 
        if self.n is None and self.mechanism.n is not None: self.n = self.mechanism.n
        
        if isinstance(self.probs,Probs):
            self._val = make_multinomial_distribution(self.n,self.probs.val)
            self._previous = None
        else:
            raise TypeError("Multinomial probs must be of type Probs")
        Param.make_initargs(self, probs = self.probs)
        self.initialized = True
        

    def roll (self):
        '''
        Sometimes we want to change the order of elements in a multinomial distribution
        but we don't want to change the statistics of the distribution.  This can 
        be done with np.shuffle but a cheaper alternative is to 'roll' the distribution.
        For small distributions (e.g. 1000) shuffling is faster by a factor of 2 but
        for large ones (e.g., 100000) rolling is ten times faster. 
        '''
        self._previous = self._val
        self._val = np.roll(self.val,rng.integers(50,500,1))
        
    def update (self,*args,roll=True,**kwds):
        update_now = self.update_when(*args,**kwds) if callable(self.update_when) else self.update_when
        if update_now:
            if self.probs.prev is None or np.all(self.probs.prev == self.probs.val):
                # probabilities haven't changed
                if roll is True: self.roll()
            else:
                # either the probs have changed or roll is False
                self._previous = self._val
                # not self.rebuild because it sets self._previous to None
                self._val = make_multinomial_distribution(self.n,self.probs.val)
            
    def rebuild (self):
        self._val = make_multinomial_distribution(self.n,self.probs.val)
        
    def describe (self,variates = 20):
        print(f"\nname: {self.name}")
        if self.mechanism is not None: print(f"mechanism: {self.mechanism.name}")
        print(f"prob: {self.probs}\nprobs.val: {self.probs.val}\nn: {self.n}")
        print(f"_val (first {variates} values): {self._val[:variates] if self._val is not None else None}")
        print(f"_previous (first {variates} values): {self._previous[:variates] if self._previous is not None else None}")
        print(f"kwds: {self.kwds}")
        if self.doc is not None: print(f"doc: {self.doc}")
        


#=============================================================================
# Array-like params : Array
#=============================================================================

class Array (Param):
    ''' Creates an np.ndarray with n rows and m columns '''
    
    @classmethod  
    def wrap_if_vec(Array,a):
        # if a is 1D, this makes it 2D because all Array methods expect > 1D
        return a if a.ndim > 1 else np.array([a])
    
    def __init__(self, *args, shape = None, **kwds):
        self.shape = shape
        super().__init__(*args,**kwds)
        self.type = 'Array'
    
    def initialize_val (self,*args,**kwds):
        if callable(self.init_val):
            self._val = Array.wrap_if_vec(self.init_val(*args,**kwds))
            if type(self._val) != np.ndarray:
                    raise ValueError("Evaluating init_val for Array must result in an Numpy array")
        
        elif type(self.init_val) == np.ndarray:
            self._val = Array.wrap_if_vec(self.init_val)
                
        elif self.shape is not None:
            # shape is specified
            if self.init_val is None:
                self._val = np.zeros(self.shape)
            elif type(self.init_val) in [int,float,bool]:
                self._val = np.full(self.shape,self.init_val)
            else:
                raise ValueError("init_val must evaluate to an int, float, bool, or an array broadcastable to dimensions {self.shape}")     
            
        else:
            raise ValueError("To initialize an Array, you must specify a shape or an array or a callable that returns an array")
        self.initialized = True
        
        # even if n and m weren't specified explicitly we can now get them from 
        # self._val, and if they  specified explicitly, the values from self._val override
        
        val_n, val_m = self._val.shape
        if self.shape is not None:
            shape_n, shape_m = self.shape
            if shape_n != val_n:
                print (f"WARNING: {self.name} has a different n than specified by `shape`")
            if shape_m != val_m:
                print (f"WARNING: {self.name} has a different m than specified by `shape`")
        
        self.n, self.m, self.shape = val_n, val_m, self._val.shape
        self._previous = None
        Param.make_initargs(self, n = self.n, m = self.m, shape = self.shape)
        
        
    def update (self, *args,**kwds):
        super().update(*args,**kwds)
        if self._val.shape != self._previous.shape:
            print (f"WARNING: updating array {self.name} changed its shape")
        
    @classmethod
    def row_normalize (Array,a):
        return a / np.sum(a, axis=1)[:,np.newaxis]
    
    @classmethod
    def row_sums (Array,a):
        return np.sum(a,axis = 1)

    @classmethod
    def col_sums (Array,a):
        return np.sum(a,axis = 0)
    
    @classmethod
    def row_means (Array,a):
        return np.mean(a,axis=1)
     
    @classmethod   
    def col_means (Array,a):
        return np.mean(a,axis=0)
    
    @classmethod
    def row_vars (Array,a):
        return np.var(a,axis=1)
    
    @classmethod
    def col_vars (Array,a):
        return np.var(a,axis=0)
    
    @classmethod
    def index2onehot (Array,index, shape):
        ''' index is an np.array of length r that holds indices between 0 and c - 1.  
        This returns an array of shape r,c that contains a one-hot encoding of the 
        column indicated by index; e.g., for c = 3 and index = np.array([0,2,1]), 
        index2onehot(index) -> [[1. 0. 0.],[0. 0. 1.],[0. 1. 0.]] '''
        zeros = np.zeros(shape)
        zeros[np.indices(index.shape)[0], index]=1
        return zeros
    
    @classmethod
    def onehot2index (Array,a):
        return np.argmax(a,axis=1)
    
    @classmethod
    def row_min_onehot (Array,a):
        ''' One-hot encoding of the column in a that holds the minimum value.  If
        two columns hold the same minimum, this takes the first.'''
        return Array.index2onehot(np.argmin(a,axis=1),a.shape)
    
    @classmethod
    def row_max_onehot (Array,a):
        ''' One-hot encoding of the column in a that holds the maximum value.  If
        two columns hold the same maximum, this takes the first.'''
        return Array.index2onehot(np.argmax(a,axis=1),a.shape)
    
    @classmethod
    def row_sample (Array, probs, one_hot = True):
        ''' probs is a 2D array in which each row is a multinomial distribution. 
        By default this returns a one-hot encoding of the column selected by 
        sampling from each row. If one_hot is not True, it returns the column index.
        
        For speed, this does not check whether the numbers in a row sum to 1.0.
        
        For machine learning purposes, this choice must run fast.  Parts of the solution are 
        https://bit.ly/3AXSWJV, https://bit.ly/3peWVzv, https://bit.ly/3G3aSq3 
        
        ''' 
        
        chosen_cols = (probs.cumsum(1) > rng.random(probs.shape[0])[:,None]).argmax(1)
        return Array.index2onehot(chosen_cols,probs.shape) if one_hot else chosen_cols
    
    
    @classmethod
    def ged (Array,p0, p1):
        ''' Generalized euclidean distance. p0 and p1 must both have the same number 
        of columns, c, and p0 must be broadcastable to p1.  Each row is treated as a 
        point in c-dimensional space. This returns an array of distances of shape r,c, 
        where r is the number of rows in p1. It uses numpy linalg.norm, which allows for 
        different distance measures than euclidean distance but one could also use 
        the more familiar np.sum((p0-p1)**2,axis=1)**.5 . '''
        
        x = np.array([p0]) if p0.ndim == 1 else p0
        y = np.array([p1]) if p1.ndim == 1 else p1
        return np.linalg.norm(x-y, axis = 1)




#=============================================================================
# Array-like params :  MultiColumn
#=============================================================================

class MultiColumn (Column):
    ''' Sometimes we want several versions of one column, such as income
    in each of four quarters of the year. This class allows us to define one 
    multicolumn whose value is an n x m array.  Its interface is identical with
    the Column interface other than having to specify which of m columns is 
    queried or assigned.
    
    The init_val of MultiColumn is processed by Array.initialize_val.
    '''
    
    def __init__(self, shape=None, **kwds):
        self.shape = shape
        super().__init__(**kwds)
        self.type = 'MultiColumn'
        
    def col (self,c):
        return self._val[:,c]
    
    '''
    Elementary comparison operators for columns. 
    
        -- For a Column c, c.eq(17) returns a boolean column where True means
           the row value is 17, i.e., c.eq(17) is equivalent to c == 17
    
        -- For a MultiColumn, c.eq(0,17) returns a boolean column for the 0th column
           of the multicolumn; i.e., c.eq(0,17) is equivalent to c[:,0] == 17
    '''
    
    def op (self,col= None,val = None,fn=None):
        if callable (val):
            return fn(self.col(col),val.__call__())
        else:
            return fn(self.col(col),val)
        
    def initialize_val (self,*args,**kwds):
        Array.initialize_val(self,*args,**kwds)
    
    def assign (self,col, val,*args,**kwds):
        self.assign_helper(val=val,col=self.col(col),*args,**kwds)
        
        
# OLD method for initializing a MultiColumn.  Might need it again one day.
# def initialize_val (self, *args, **kwds):  
         
#         if callable(self.init_val):
#             v = self.init_val(*args,**kwds)
#         else:
#             v = self.init_val
#             print(type(v))
        
#         self._val = np.zeros((self.n,self.m)).astype(self.element_type(v))  
        
#         if type(v) in [int,bool,float]:     
#             self._val[:] = v
#         elif type(v) == np.ndarray:
#             try: 
#                 # try to broadcast v to _val; it'll work even if v is a row of length m
#                 self._val[:] = v
#             except:
#                 # but perhaps v is a single column; it needs an extra axis to broadcast to _val
#                 try: 
#                     self._val[:] = v[:,np.newaxis]
#                 except:
#                     raise ValueError(f"init_val cannot be broadcast to an array of shape {self._val.shape}")
#         self._previous = None


#=============================================================================
# Column-like params : Simplexes
#=============================================================================

class Simplex (Param):
    ''' 
    A standard simplex or probability simplex is a space in which each point is 
    a multinomial; that is, the coordinates of the point sum to 1.0. If there are
    two classes in the multinomial then the simplex is line; if three, a triangle, 
    and so on. 
    
    The Simplex class methods deal with setting up the simplex and dynamically
    changing its attributes. 
    
    The init_val of Simplex is processed by Array.initialize_val.
    
    The Simplex `update` method moves points in the simplex. The coordinates of
    points can be masked, which means they are not changed by the `update` method.
    For example, masking the first coordinate of the point [.5,.2,.3] prevents
    the `update` method from moving the point along the first dimension. Any 
    movement would be effected by changing the remaining coordinates, but because
    the coordinates must sum to 1.0, there's only one degree of freedom; the point
    must move in one dimension.  Clearly, if a point has k coordinates it is
    rendered immobile by masking k-2 or more of its coordinates.
    
    Masks can be dynamic.
    
    
    '''
    
    def __init__(self, *args, shape=None, mask = None, **kwds): 
        self.shape = shape
        self.mask = mask
        super().__init__(**kwds)
        self.type = 'Simplex'
        
        
    def initialize_val (self, *args, **kwds):
        Array.initialize_val(self, *args, **kwds)
        if self.mask is not None:
            if self.mask.shape != self._val.shape:
                raise ValueError("A mask must have the same shape as the simplex it masks")
        super().make_initargs(mask = self.mask)
        self.initialized = True
    
    def update (self, destination, rate, selected = True):
        ''' 
        This changes the probability vectors of selected rows in a Simplex,
        which amounts to moving selected points in the Simplex. Only `selected` 
        points move.
        
        First this makes a copy of self.val called A.  D is the destination 
        locations. M is a mask.  D can have the same shape as A, in which case 
        each point in A moves toward the corresponding point in D; or D can be
        a single point, in which case all points in A move toward it.
        
        R can be a single rate (a scalar) or an array of rates of length self.n
        
        The final distance from A to D is determined by R.  If R = 0 then A remains
        where it was, if R = 1 then A moves all the way to D. 
        
        If some probabilities in a point are masked and we move the remaining 
        lower-dimensional points, then their new unmasked probabilities 
        plus those of their masked probabilities can sum to more than 1.0. 
        To ensure that moves with masked probabilities remain in the simplex, 
        we first rescale the unmasked probabilities, then move their points, 
        then invert the rescaling. '''
        
        update_now = self.update_when(*args,**kwds) if callable(self.update_when) else self.update_when
        if update_now:
            
            self._previous = copy.copy(self.val) 
            A = copy.copy(self.val) 
            # destination might be an Array or Simplex or Probs
            D = destination.val if isinstance(destination,Param) else destination
            # rate might be a Column or simple Param
            R = rate.val if isinstance(rate,Param) else rate 
            
            if self.mask is None:
                # no rescaling needed
                A = (1 - R) * A  + R * D
            else:
                M = self.mask.val if isinstance(self.mask,Param) else self.mask
                UM = ~M
                p0_u = D * UM
                p0_u_rescaled = p0_u / np.sum(p0_u,axis = 1)[:,np.newaxis]
                
                p1_u = A * UM
                p1_sums = np.sum(p1_u,axis=1)[:,np.newaxis]
                p1_u_rescaled = p1_u/p1_sums
                
                A[UM] = (p1_sums * ((1 - R) * p1_u_rescaled  + R * p0_u_rescaled))[UM] 
            
            
            self._val[selected] = A[selected]
    
    
    def row_renorm_unmasked (self, normalize_fully_masked = True, warn = False):
        y = copy.copy(self._val)
        M = self.mask.val if isinstance(self.mask,Param) else self.mask
        # excess mass by row
        e = (np.sum(y,axis=1)-1)[:,np.newaxis]
        # some rows have all values masked
        m_all = np.where(np.all(M,axis=1))[0]
        if any(m_all):
            y1 = y[m_all,:]
            if normalize_fully_masked:
                if warn:
                    print(f"Rows\n{m_all}\n{y1}\nwere fully masked and had to be renormalized")
                y1 = y1 / np.sum(y1,axis=1)[:,np.newaxis]
                y[m_all] = y1
            else:
                raise ValueError(f"\nRows\n{m_all}\nValues\n{y1}\nsummed to more than 1 and were fully masked and normalize_fully_masked is False")
        
        m_nall = np.where(~np.all(M,axis=1))[0]
        if any(m_nall):
            y1 = y[m_nall,:]
            m1 = M[m_nall,:]
            e1 = e[m_nall,:]
            
            a0 = y1 * m1.astype(int)
            b0 = np.sum(a0,axis = 1) [:,np.newaxis]
            a1 = y1 * (~m1).astype(int)
            b1 = np.sum(a1,axis = 1) [:,np.newaxis]
            
            g = b0>1
            if any(g):
                raise ValueError(f"ValueError: Masked values in row {list(np.where(g)[0])} sum to more than 1.0.\nsum of masked:\n{b0}\n")
                # print(f"available mass:\n{1 - b0}\n")
                # print(f"available probs:\n{a1}")
                # print(f"adjustment:\n{a1/b1 * e1}\n")
                
            f = (y1 - (a1/b1 * e1) )
            y[m_nall] = f
            
        # print(y)
        # print(np.sum(y,axis=1))
        return y
    
    def sample (self, index=True, unmasked_only = False):
        ''' Sampling from val returns a one-hot encoding of the selected option.  
        If an index representation  is preferred (i.e., an integer that represents
        which option is sampled, then specify form = 'index'. 
        
        Depending on the values in masked columns, a masked option might be sampled.  
        For some models this makes sense, for others it defeats the purpose of 
        masking.  The parameter 'unmasked_only' restricts decisions to unmasked 
        columns.
        '''
        
        if unmasked_only:
            unmasked_vals = self.val * (~self.mask).astype(int)
            renormalized = unmasked_vals / np.sum(unmasked_vals,axis=1)[:,np.newaxis]
            s = Array.row_sample(renormalized)
            
        else:
            s = Array.row_sample(self.val)
        return Array.onehot2index(s) if index else s
    


#=============================================================================
# Counters
#=============================================================================

class Counter (Param):
    def __init__(self, increment = 1, **kwds):
        self.increment = increment
        super().__init__(**kwds)
        self.type = 'Counter'
        self.dynamic = True  
    
    def initialize_val (self):
        # hack because can't have a default init_val here if we use super().init
        if self.init_val is None: self.init_val = 0
        super().initialize_val()
        super().make_initargs(increment = self.increment)
        self.initialized = True
       
    def update (self,*args,**kwds):
        update_now = self.update_when(*args,**kwds) if callable(self.update_when) else self.update_when
        if update_now:
            self._previous = self._val
            self._val =  self.val + self.increment
        
    def describe (self):
        Param.describe(self)
        print(f"start: {self.init_val}\nincrement: {self.increment}")



#=============================================================================
# Switches
#=============================================================================

class Switch (Param):
    ''' A Switch can be in one of n states depending on logical conditions, which
    may include the values of other parameters.  Switches can be expensive to
    evaluate so there are two optimizations.  First, the switch checks the condition
    of the state it's currently in before checking any other condition.  Second
    the switch inherits from DynamicParam an `update_when` condition that must be 
    true for it to check any individual conditions. '''
            
    def __init__(self, conditions = None, labels= None,  **kwds):
        
        # all the conditions in a switch should be callable
        if all([callable(c) for c in conditions]):
            self.conditions = [MethodType(c,self) for c in conditions]
        else:
            raise TypeError("All the conditions in a Switch should be callable")
        
        self.labels = np.arange(len(conditions)) if labels is None else labels      
        
        super().__init__(**kwds)
        self.type='Switch'
        self.dynamic = True
        
        # store the code strings for the conditions
        self.code4callables.update(
            {'conditions': " ".join([inspect.getsource(c) for c in conditions if callable(c)])})
        
        
    def initialize_val (self,*args,**kwds):
        # self._val will hold the index of the state that the Switch is in
        self._val = self.which_condition(*self.args,**self.kwds)
        self._previous = None
        super().make_initargs(conditions=self.conditions,labels=self.labels)
        self.initialized = True
            
    def which_condition (self,*args,**kwds):
        ''' Loops through the conditions and returns the labels of all that
        are satisfied.  Returns None if none is. '''
        return [label for cond,label in zip(self.conditions,self.labels) if cond(*args,**kwds)]
    
    def update (self,*args,**kwds):
        update_now = self.update_when(*args,**kwds) if callable(self.update_when) else self.update_when
        if update_now:
            self._val = self.which_condition(*args,**kwds)
                
    def rebuild (self,*args,**kwds):
        self._val = self.which_condition(*args,**kwds)
        self.initargs['_val'] = self._val
        self.initargs['_previous'] = None
        
    def describe (self):
        if self.name is not None:
            print(f"\nname: {self.name}")
        if self.mechanism is not None:
            print(f"mechanism: {self.mechanism.name}")
        print(f"current condition: {self.curr}")
        print(f"all labels: {self.labels}")
        print(f"update_when method:\n   {self.update_when}")
        print(f"conditions:")
        for cond in self.conditions: print(f"   {cond}")
        print(f"kwds: {self.kwds}")
        if self.doc is not None: print(f"doc: {self.doc}")
                


               


#=============================================================================
# Cycles
#=============================================================================

class Cycle (Param):
    ''' A Cycle has a list of values that succeed each other in a fixed order.  
    The update_fun must evaluate whether it's time to advance the cycle. It must
    return a boolean. 
    ''' 
    def __init__(self, labels = None, **kwds):
        self.labels = np.array(labels)
        super().__init__(**kwds)
        self.type='Cycle'
        self.dynamic = True
              
    def label_to_front (self,label):
        ''' rolls labels until labels[0] is the specified label'''  
        while self.labels[0] != label:
            self.labels = np.roll(self.labels,shift = -1)
        
    def initialize_val (self,*args,**kwds):
        start_label = self.init_val(*args,**kwds) if callable(self.init_val) else self.init_val
        if start_label is None: start_label = self.labels[0]
        if start_label in self.labels:
            self.label_to_front(start_label)
        else:
            raise ValueError(f"label {start_label} is not in labels {self.labels}")
        self._val = self.labels[0]
        self._previous = self.labels[-1]
        super().make_initargs(labels=self.labels)
        self.initialized = True
        
    @property
    def lval(self):
        if self._val is None:
            print(f"lazy evaluation of {self.name} _val")
            self.initialize_val(*self.args,**self.kwds)
        return self.labels[0]
        
    @property
    def val(self):
        return self.labels[0]
    
    @val.setter
    def val(self, label):
        self._previous = self._val
        self.label_to_front(label)
        self._val = self.labels[0]

    @val.deleter
    def val(self):
        del self._val
    
    @property
    def curr(self):
        return self.labels[0]
    
    @property
    def prev(self):
        return self.labels[-1]
    
    @property
    def next(self):
        return self.labels[1]
    
    
    def update (self,*args,**kwds):
        ''' Unlike in other Params, the update function does not set self._val. 
        Its purpose is to evaluate whether it is time to advance the cycle. It
        must return a boolean.'''
        update_now = self.update_when(*args,**kwds) if callable(self.update_when) else self.update_when
        if update_now:
            self._previous = self._val
            self.labels = np.roll(self.labels,shift = -1)
            self._val = self.labels[0]
            
    def reset (self):
        self.labels = self.initargs.get('labels')
        self._val = self.labels[0]
        self._previous = None
        
    def rebuild(self,*args,**kwds):
        ''' rebuild cannot give a different result than reset unless init_val is callable ''' 
        if callable(self.init_val):
            self.initialize_val(*args,**kwds)
        else:
            self.reset()
        
        
    def describe(self):
        print(f"\nname: {self.name}")
        if self.mechanism is not None:
            print(f"mechanism: {self.mechanism.name}")
        print(f"labels: {self.labels},\ncurrent label: {self.curr}\nstart label: {self.initargs['labels'][0]}")
        print(f"update_val: {self.update_val}\nupdate_when: {self.update_when}")
        print(f"kwds: {self.kwds}")
        if self.doc is not None: print(f"doc: {self.doc}")
        






  