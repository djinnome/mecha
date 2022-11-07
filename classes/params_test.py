#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:54:47 2022

@author: prcohen
"""

import numpy as np
from numpy.random import default_rng
# stuff for random sampling, note that a random seed is given for replicability
rng = default_rng(1108) 

import os
os.chdir("/Users/prcohen/Pitt/Code/Mechanisms")
from mecha.classes.param_classes import (
    Param,Probs, Column, Cohort, MultiColumn, Multinomial, Simplex, Array, Switch, Cycle, Counter
    )

import mecha.classes.param_classes as pc


##############################################################################
### Testing functions of all Params - init_val, initargs, assignment, reset, rebuild
##############################################################################

p = Param(
    name = 'p',
    init_val = np.round(rng.random(),4),
    doc = 'Illustrating reset'
    )

print(f"val:{p.val}, curr: {p.curr},  previous: {p._previous}")
p.val = 100
assert (p.val == 100)

print(f"val:{p.val}, curr: {p.curr},  previous: {p._previous}")
p.reset()
print(f"val:{p.val}, curr: {p.curr},  previous: {p._previous}\n")

assert(p.val == 0.431)


print(f"val:{p.val}: {p._previous}")
p.reset()
print(f"val:{p.val}; Same values as before\n")

q = Param(
    init_val = lambda cls: rng.integers(2,6,1),
    doc = 'This parameter will be passed to another parameter'
    )

print (f"q.val {q.val}")

p = Param(
    name = 'p',
    init_val = lambda self: np.round(rng.random(q.val),4),
    )

print (f"p.val {p.val}\n")
assert (np.all(p.val == [0.1938, 0.6678]))

q.rebuild()
print(f"q now has val {q.val}")
p.rebuild()
print(f"p now has {q.val} values")

assert(len(p.val) == 4)

##  reset relies on `initargs`, which are stored when the param is created
##  However, reset makes a copy of initargs, as can be seen by comparing
## memory locations:
    
p = Param(init_val = np.arange(10))
print(f"initial value\n{p.val}\ninitargs:\n{p.initargs}\n")
p.reset()
print(f"initial value\n{p.val}\ninitargs:\n{p.initargs.get('init_val')}\n")

print(f"Same memory location? {id(p.val) == id(p.initargs.get('init_val'))}")



##############################################################################
### Testing functions of all Params - names, anonymous params
##############################################################################

print(f"\nThe parameter assigned to the variable p was given the name {p.name}")
print(f"The 'anonymous' parameter assigned to the variable q has name {q.name}\n")
assert(q.name == 'anon_101')



##############################################################################
### Testing functions of all Params - update_val and update_when
##############################################################################

# An update function can be any value or function or lambda expression 

p = Param(
    name = 'p',
    init_val = 3,
    update_val = lambda cls: cls.val + rng.integers(0,100,1)[0]
    )

print(f"val: {p.val}, curr: {p.curr}, prev: {p.prev}")
assert(p.val == 3)

p.update()
print(f"val: {p.val}, curr: {p.curr}, prev: {p.prev}\n")
assert(p.val == 76)


# update_val can take argumements 
p = Param(
    name='p',
    init_val = lambda self: rng.integers(0,10,5), 
    update_val = lambda self,x: rng.integers(0,x,5)
    )

print(f"initial value: {p.val}")
p.update(x=100)
print(f"after update:  {p.val}\n")

assert(p.val[0] == 55)
assert(p.prev[0]==5)


# update_when controls when updating happens

p = Param(
    init_val = rng.integers(0,10,1), 
    update_val = lambda self, x, **kwds: rng.integers(0, 10, x),
    update_when = lambda self, day, **kwds: day in [3,6,9],
    useless_param = 3
    )

print(f"initial value: {p.val}")
for i in range(10):
    p.update(day = i, x = i)
    print(f"day: {i} p.val: {p.val}")

print()

assert(len(p.val == 9))


##############################################################################
### Testing Probs
##############################################################################

def get_probs (cls):
    p = rng.random()
    return [p,1-p]

p = Probs(
    init_val = get_probs
    )

print(f"p.val:  {p.val}")
p.val = [.5, .5]
print(f"p.val:  {p.val}")
p.reset()
print(f"p.val:  {p.val}")
p.rebuild()
print(f"p.val:  {p.val}\n")

try:
    q = Probs(
        name = 'p',
        init_val = [.7,.31]
        )
except(Exception) as e:
      print(e)
      
print()
      

# A slightly more interesting example:  On every iteration the population gets
# a year older but no-one is born. We'll define a prob equal to the fraction
# of the population older than a given age.


def older_than(age):
    prob = np.sum(pop > age) / len(pop)
    return [prob,1-prob]

pop = rng.integers(0,80,1000)
age = 30

p = Probs(
    name='p',
    init_val = older_than(age), 
    update_val = lambda cls, age: older_than(age),
    )

for i in range(10):
    pop = pop + 1
    p.update(age=age)
    print(p.val)

assert(p.val == [0.727, 0.273])
assert(np.all(np.round(p.prev,4) == [0.717, 0.283]))

print()


##############################################################################
### Testing Columns
##############################################################################

p = Column(n=20,init_val = 0)
print(f"initial value:\n{p.val}")
p._val[0:10]=1
print(f"after assiging 1 to the first ten rows:\n{p._val}\n")

c = Column(
    name = 'c',
    init_val = lambda self, **kwds: rng.integers(0,5,self.n),
    selected = np.array([1,1,1,0,0,0,0,0,0,0]).astype(bool),
    n=10
    )

print(f"val {c.val}, prev: {c.prev}")
c.rebuild()
print(f"val {c.val}, prev: {c.prev}")
c.assign(rng.integers(5,10,10))
print(f"val {c.val}, prev: {c.prev}")
print(f"c.gt(7) = {c.gt(7)}\n")

assert(c.gt(7)[0] == True)
assert(np.sum(c.gt(7)) == 5)


# make a column s that selects rows in another column c

n = 30
val = 5

c = Column(init_val = rng.integers(0,10,n))
c.initialize_val()

s = Column(
    # make a boolean column where True means a row of c contains a value <= val
    init_val = c.le(val), 
    # these selected rows can change
    update_val = lambda self,col,val: col.le(val)
    )

    
for i in range(5):
    print(f"selecting c<={i}:")
    # change the selected rows, print c and s
    s.update(col = c, val = i)
    print(c.val)
    print(s.val.astype(int))

assert (np.sum(s.val) == 16)
print()

##############################################################################
### Testing Cohorts
##############################################################################

# Cohorts are boolean columns that are used to select rows in other columns

n = 1000
c = rng.integers(0,10,n)
p = Param(init_val = 3)
p.initialize_val()

# h has length n and True in each row for which c < p.val

h = Cohort(n=n,
            init_val = lambda self: c < p.val
            )

h.initialize_val()

print(f"The number of True values in h.val is {h.size}")  # should be near 300
assert(h.size == 304)



p.val = 7
print(f"The number of True values in h.val is {h.size}")   # h isn't updated yet, so no change. 
h.rebuild()
print(f"The number of True values in h.val is {h.size}")   # should be near 700
assert(h.size == 685)

p.val = 1
h.rebuild()
print(f"With p.val == 1, these are the cohort members:\n{h.members}\n\n")

##############################################################################
### Testing MultiColumns
##############################################################################


mc = MultiColumn(
    shape = (10,4),
    init_val = lambda self: rng.integers(0,7,(self.shape[0],self.shape[1]))
    )


print(f"A small MultiColumn param with 10 rows and 4 columns: \n{mc.val}\n")

print(f"The first column in the multicolumn:\n{mc.val[:,0]}")
print(f"Using a comparator; the first argument is desired column:\n{mc.le(0,4).astype(int)}")
mc.assign(0,9,selected=mc.le(0,4))
print(f"Assignment, again the first argument is desired column:\n{mc.val[:,0]}")


mc.assign(0,17,selected = rng.random(mc.n) > .5)
print(f"after setting a random selection of these to 17:\n{mc.val}\n")

# # you don't have to use assign; selected can be a tuple of indices
mc.val[(4,5),0]=12
print(f"after setting two more indices of column 0 to 12:\n{mc.val}\n")

assert(np.sum(mc.col(0)) == 117)


##############################################################################
### Testing Multinomials
##############################################################################

m = Multinomial(
    probs = Probs(init_val = np.array([.3,.1,.2,.4])), 
    n = 100,
    doc = 'a multinomial'
    )

m.probs.initialize_val()
m.initialize_val()

print(f"\ncurrent:\n{m.curr}")
print(f"number of zeros: {np.sum(m.eq(0))}")
m.roll()
print(f"\nafter roll:\n{m.curr}")
print(f"number of zeros: {np.sum(m.eq(0))}")
print(f"\nprevious:\n{m.prev}")
m.reset()
print(f"\nafter reset:\n{m.curr}")
m.rebuild()
print(f"\nafter rebuild:\n{m.curr}")
print()


m = Multinomial(
    probs = Probs(
        init_val = rng.dirichlet((1,1,1),1)[0],
        # returns a different probability distribution over three classes
        update_val = lambda self: rng.dirichlet((1,1,1),1)[0]
        ), 
    n = 1000
    )

    
for i in range(6):
    m.probs.update() # update the underlying probs
    m.update(i)      # and then update the Multinomial
    u,c = np.unique(m.val,return_counts = True)
    print(f"probs: {[np.round(p,4) for p in m.probs.val]}\t\tcounts: {c}")
    

assert(np.all(c == [ 20, 356, 624]))


##############################################################################
### Testing Arrays
##############################################################################


a = Array(shape = (3,5))
print(a.val)

a = Array(shape = (3,5), init_val = 3)
print(a.val)

a = Array(shape = (3,5), init_val = True)
print(a.val)

a = Array(shape = (3,5), init_val = rng.integers(0,5,(7,3)))
print(a.val)

a = Array(shape = (3,5), init_val = lambda self: rng.integers(0,5,5))
print(a.val)

a = Array(shape = (3,5), init_val = lambda self: rng.integers(0,5,(3,5)))
print(a.val)

a = Array(init_val = rng.integers(0,5,(5)))
print(a.val)

a = Array(init_val = rng.integers(0,5,(3,5)))
print(a.val)

# updating issues a warning if it changes the shape of the array

print()
a = Array(
    init_val = np.zeros((3,5)),
    update_val = np.ones((2,5))
    )

print()
a.initialize_val()
a.update()
print()

a = Array(init_val = lambda self: rng.integers(0,5,(3,5)))
print(a.val)



# Arrays have many class methods, only some of which are tested here
ged = Array.ged(a.val[0],a.val[1])
print(f"Generalized Euclidean distance between first two rows of a: {ged}\n")
assert(ged[0] == 6.855654600401044)

ged = Array.ged(a.val[0],a.val)
print(f"ged between first row and all rows of a: {ged}\n")
assert(ged[0]==0)

print(f"One-hot encoding of maximum ged value: {Array.row_max_onehot(np.array([ged]))}\n")
print(f"Index of maximum ged value: {Array.onehot2index(Array.row_max_onehot(np.array([ged])))}\n")

probs = rng.dirichlet([1,1,1,1,1],10)
print(f"Probability vectors:\n{probs}")

sample = Array.row_sample(probs)
print(f"One-hot encoding of the selected column sampled from each row\n{sample}\n")

sample = Array.row_sample(probs)
print(f"Sampling again gives a different result\n{sample}\n\n")


##############################################################################
### Testing Simplexes
##############################################################################

# Simplexes are themselves straightforward: they are just arrays of probability
# vectors.  But probabilities can be masked, and masks can be changed, so testing
# here is done with tiny simplexes that contain only four points.

x = np.array(
    [[.5,.2,.3],
      [.1,.2,.7],
      [.3,.4,.3],
      [.25,.25,.5]
      ])

m = np.array(
    [[0,0,0],
      [0,0,1],
      [0,1,1],
      [0,1,0]
      ]).astype(bool)


S = Simplex(
    init_val = x,
    mask = m
    )

# If the probability vectors in a simplex are of length k, then sampling from
# a simplex will return k indices of sampled values. For example, with four 
# points in a simplex, each of which contains three probabilities, one sample
# will return 0, 1 or 2 for each of four points.  

# Consider the first point in the simplex, with probabilities [.5,.2,.3].  If
# we sample from these probabilities 10000 times, we'd expect the first column
# to be selected 5000 times, the second 2000 times and the thrd 3000 times:

samples = np.stack([S.sample() for i in range(10000)],axis=0)    
print(f"Number of times each column was selected when sampling from the first point:\n{[np.sum(samples[:,0]==i) for i in [0,1,2]]}")
print(f"And for sampling from the second point:\n{[np.sum(samples[:,1]==i) for i in [0,1,2]]}\n")

assert(np.all([np.sum(samples[:,1]==i) for i in [0,1,2]] == [982, 1978, 7040]))

# # However, these numbers change dramatically if we sample only *unmasked* points:

samples = np.stack([S.sample(unmasked_only=True) for i in range(10000)],axis=0)    
print(f"Sampling from the first point, unmasked only:\n{[np.sum(samples[:,0]==i) for i in [0,1,2]]}")
print(f"Sampling from the second point, unmasked only:\n{[np.sum(samples[:,1]==i) for i in [0,1,2]]}\n")

## The frequencies for the first point are very little changed because all the 
# probabilities in the first point are unmasked, but in the second point the last
# probability is unmasked and the second is twice as big as the first, so we 
# expect 3333 and 6666 as the freqencies for sampling colums 0 and 1.

## If we change the mask, these frequencies change:
     
n = np.array(
    [[0,0,1],
     [1,0,1],
     [1,0,1],
     [1,1,0]
      ]).astype(bool)

S.mask = n

samples = np.stack([S.sample(unmasked_only=True) for i in range(10000)],axis=0)    
print(f"After changing mask:\nSampling from the first point, unmasked only:\n{[np.sum(samples[:,0]==i) for i in [0,1,2]]}")
print(f"Sampling from the second point, unmasked only:\n{[np.sum(samples[:,1]==i) for i in [0,1,2]]}\n")

    

##############################################################################
### Testing Switches
##############################################################################

c = Counter(init_val = 0,increment = 7)

s = Switch(
    conditions = [
        lambda self: c.val%2 == 0,
        lambda self: c.val%3 == 0,
        lambda self: c.val%5 == 0,
        ],
    labels = ['div_by_2','div_by_3','div_by_5']
    )


for i in range(10):
    print(c.val,s.val)
    c.update()
    s.update()
    
    
    
print()





##############################################################################
### Testing Counters
##############################################################################

c = Counter()
print(f"c initial value {c.val}, c increment {c.increment}")


# update_when says no more updates after c.curr > 23
c = Counter(
    init_val = 20,
    # don't update once self.val has reached a maximum
    update_when = lambda self: self.val <= 23)

for i in range(8):
    print(f"current:{c.curr}  previous: {c.prev}")
    c.update()

print()

c.val = 10
c.increment = 3
for i in range(10):
    print(f"current:{c.curr}  previous: {c.prev}")
    c.update()

assert(c.prev == 22)
print("\n\n")


##############################################################################
### Testing Cycles
##############################################################################

c = Cycle(
    labels = ['winter','spring','summer','autumn'],
    init_val = 'summer'
    )

print(f"Cycling through seasons beginning with summer; cols are val, curr, prev:")
for i in range(6):
    print(c.val,c.curr,c.prev)
    c.update()

print()

c.val = 'winter'
print(f"After assigning c.val = {c.val} the labels 'cycle' to {c.labels}\n")
c.reset()
print(f"After resetting c.val the labels 'cycle' to {c.labels}\n")



def random_label (cls,**kwds):
    return rng.choice(['winter','spring','summer','autumn'])

c = Cycle(
    labels = ['winter','spring','summer','autumn'],
    init_val = random_label,
    update_when = lambda self, date: date in self.kwds.get('change_season'), # Try setting update_when = True
    change_season = [5,10,15,20]
    )

print(c.val)
print(c.labels)


c.reset()
print(f"after reset: current label: {c.curr}")
for i in range(5):
    c.rebuild()
    print(f"after rebuild: current label: {c.curr}")


print()
for i in range(19):
    c.update(date=i)
    print(f"season:  {c.curr}")

print("\n\n")



##############################################################################
### Testing Rules
##############################################################################

from mecha.classes.rule_classes import Rule
n = 20

what_to_plant = Column(
    n = n,
    init_val = np.zeros(n)
    )

ready_to_plant = Cohort(
    n=n,
    init_val = np.full(n,False),
    update_val = lambda self: (rng.random(n) > .7) & (planted.val == False)
    )

planted = Cohort(
    n=n,
    init_val = np.full(n, False)
    )

what_to_plant.initialize_val()
ready_to_plant.initialize_val()
planted.initialize_val()

def yet_to_plant (self):
    x = any(what_to_plant.val == 0)
    if x is False: print("everyone has planted")
    return x
    

plant = Rule(
    n = n,
    condition = yet_to_plant,
    cohort = ready_to_plant,
    actionlists = [[(what_to_plant, 1),(planted, True)],
                   [(what_to_plant, 2),(planted, True)],
                   [(what_to_plant, 3),(planted, True)] ] ,
    probs = Probs(init_val = np.array([.2,.3,.5]))
    )

plant.initialize_val()


# Number ready to plant will not be non-decreasing because we are not yet updating planted
for i in range(10):
    ready_to_plant.update()
    print(f"Number ready to plant: {np.sum(ready_to_plant.val)}")
print()
                 
    
for i in range(15):
    ready_to_plant.update()
    plant.run_rule()  
    print(f"number who have planted: {np.sum(planted.val)}")
    print(f"what they planted: {what_to_plant.val}\n")
    
#assert(what_to_plant.val[0] == 3 and what_to_plant.val[1] == 3)
    
    
    
## Simple example of running a rule with only one actionlist (i.e. without probs)    
n = 30    

c = pc.Column(n=n, init_val = lambda self: rng.integers(0,10,n))
           
r = Rule(
    n = n,
    condition = True, 
    cohort = pc.Cohort(n=n,init_val = lambda self: c.val > 4),
    actionlists = [(c,6)]
    )

print(f"The column before running the rule:\n{c.val}\n")
r.run_rule() # all values of c greater than 4 should be replaced by 6
print(f"The column after running the rule:\n{c.val}\nValues greater than 4 should be replaced by 6\n")



# conditional execution of the rule
n = 1000   

c = pc.Column(n=n, init_val = lambda self: rng.integers(0,10,n))
           

# The rule will convert 30% of values > 4 into 6, but only when x > 3. So the 
# number of 6's remains constant until x > 3 and then increases:
    
r = Rule(
    n = n,
    condition = lambda self,x: x > 3, 
    cohort = pc.Cohort(n=n,init_val = lambda self: (c.val > 4) & (rng.random(n)>.7)),
    actionlists = [(c,6)],
    )

for i in range(10):
    r.cohort.rebuild()
    r.run_rule(x=i)
    print(i, np.sum(c.val==6))


# '''
# Now try it with probs.  We'll use n = 100000, so the cohort should number 
# roughly 50000.  The first and second action lists set cohort rows to 10
# and 11 respectively, the third is a "do nothing" action.  These action lists
# have probabilities .6, .3 and .1, respectively.  We expect the final counts
# in c to be approximately 10000 for values 0,1,2,3,4.  Of the roughly 50000
# values greater than 4, we "do nothing" to ten percent of them, so these rows 
# retain their original values.  Thus the counts of 5,6,7,8,9 should be 
# approximately 1000.  We expect the counts of 10 and 11 to be .6 * 50000 = 30000
# and .3 * 50000 = 15000, respectively.
# '''

n = 100000  
c = pc.Column(n=n, init_val = lambda self: rng.integers(0,10,n))

print(f"Column frequencies at the outset\n{np.unique(c.val,return_counts=True)}")
           
r = Rule(
    n = n,
    condition = True, 
    cohort = pc.Cohort(n=n,init_val = lambda self: c.val > 4),
    actionlists = [[(c,10)],
                    [(c,11)],
                    []
                    ],
    probs = pc.Probs(init_val=[.6,.3,.1])
    )

r.run_rule() 
print(f"Column frequencies after running rule\n{np.unique(c.val,return_counts=True)}\n")
c.rebuild()
r.run_rule() 
print(f"Column frequencies after rebuilding column and running rule\n{np.unique(c.val,return_counts=True)}\n")
print(f"These are wrong because we didn't rebuild the cohort\n")
c.rebuild()
r.cohort.rebuild()
r.run_rule() 
print(f"Column frequencies after rebuilding column and cohort and running rule\n{np.unique(c.val,return_counts=True)}\n")
    

###############################################################################
# Testing Influences
###############################################################################   

from mecha.classes.rule_classes import Influence
from mecha.classes.utils import plot_ternary

# The following illustrates how an influence works.  A key point is that we allow
# masking but not clamping of probabilities, and I choose `small val` such that 
# no point in the simplex can have more than k-1 coordinates masked.  This matters
# because masking all the coordinates breaks things.  


n = 300
m = 3

# We'll initialize the Simplex with values from an Array, and we'll initialize
# the Array with a sample from a Dirichelet distribution.  This isn't necessary:
# we could initialize the Simplex with an np.array.

init_probs = Array(
    shape = (n,m),
    init_val = rng.dirichlet(np.full(m,1),n)
    )

# We also will define a mask.  Influences cannot change masked coordinates.  If points
# have 3 dimensions, as here, then masking one coordinate means the point moves on a
# 2D shape (i.e., a line) and masking two coordinates means the point cannot move at all.
# We'll mask all the coordinates that are smaller than `small_val`.  Why? No reason, it
# was just easy to do for this example. 

small_val = .1

mask = Array(
    shape=(n,m),
    init_val = init_probs.val < small_val
    )

# Now we define the simplex and plot it showing the masked (green) and unmasked (red) points

S = Simplex(
    init_val = init_probs.val,
    mask = mask
    )

plot_ternary(vertex_labels = ['a','b','c'], points=S.val, color_by = np.any(mask.val,axis=1))

# A new functionality for Influences is that they can have Cohorts and Conditions and
# multiple actionlists.  Just like Rules!  What you'll see when you run the Rule is
# destinations (i.e., two actionlists) randomly assigned to population members. Points
# will move to one or the other destination and because Influences fo *not* update
# the underlying multinomial, a point will always move toward one destination. Note
# that the green points moe toward these two destinations as well, but they can only
# move along one dimension because of masking.


# Also, each point can have its own rate. Here we define two rates, 0.1 and 0.3, 
# assigned randomly to points, so some move three times faster than others.  I wanted 
# this capability to model people being fickle or stubborn in the face of influences. 
 
rand = (rng.random(n) > .5)
rate = np.where(rand,.1,.3)[:,np.newaxis]

    
i = Influence(
    n = S.n,
    simplex = S,
    condition = True,
    cohort = True, 
    # Try it with a Cohort! Cohort(n = S.n, init_val = lambda self: S.val[:,0] > .2),
    actionlists = [[(S, Probs(init_val = np.array([1/3,1/3,1/3])), rate)],
                    [(S, Probs(init_val = np.array([.7,.2,.1])), rate)]
                    ],
    probs = Probs(init_val = [.5,.5])
    )
    

# Notice that the green points move along a single dimension because one dimension is
# masked, whereas the red points have three unmasked coordinates and so can move in 2D.
# The green points will "squeeze together" along one dimension and end up "pointing to"
# the red clusters! All except for the green points near each vertex.  These cannot 
# move at all because they have two masked coordinates.  

for x in range(42):
    i.run_rule()  
    if x%6 == 0:
        plot_ternary(vertex_labels = ['a','b','c'], points=S.val,color_by = np.any(mask.val,axis=1))   
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
