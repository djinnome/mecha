#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:18:59 2022

@author: prcohen
"""
import os, time
os.chdir("/Users/zuck016/Projects/Proposals/ASKEM")
from mecha.classes.registry import Registry
from mecha.classes.mechanism_classes import MechMap

import mecha.mechanisms.y as y
import mecha.mechanisms.x as x


mx = x.m_x
my = y.m_y


'''
When all modules are loaded, there are three ways to initialize un-initialized 
params:
    
    -- run initialize_val() for each such param
    -- run the initialize method for each mechanism
    -- run Registry.initialize_all_mechanisms
    
The first two methods *may* fail depending on the order in which they are called.
The last method should never fail because it is a "two-pass" initializer.
'''

print(mx.x0.val)
print(my.y0.val)

Registry.initialize_all_mechanisms()

d,g = MechMap.make(x.m_x,y.m_y,viz=True)
display(g)

assert(my.y0.val == 51)
assert(mx.x0.val == 153)

# my.y0.update()
# assert(my.y0.val == 153)

# mx.x0.update()
# assert(mx.x0.val == 27)

