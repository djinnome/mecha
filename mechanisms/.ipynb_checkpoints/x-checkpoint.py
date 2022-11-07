#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 11:48:02 2022

@author: prcohen
"""

import os, time
os.chdir("/Users/zuck016/Projects/Proposals/ASKEM")

from mecha.classes.registry import Registry
from mecha.classes.param_classes import Param
from mecha.classes.mechanism_classes import Mechanism

m_x = Mechanism(name = 'm_x', n = 20)
current_mechanism = m_x


'''
This silly example says the init_val of x0 will be m = 3 times the value of the
y0 param in a mechanism called m_y.  m_y is defined in a module called y, but 
while the mechanism matters, the module in which it is defined does not. This
is because mechanisms register their params and register themselves with Registry.
The Registry `get` method takes two string arguments -- the name of a mechanism
and the name of a param -- and returns the param object if it exists. 

However, it won't necessarily exist because modules may be loaded in any order.  
So we say init_now = False, which tells mecha to not run initialize_val.
'''

x0 = Param(
    name = 'x0', 
    init_now =False, 
    init_val = lambda self: 3 * Registry.get('m_y','y0').val, 
    update_val = 27
    )

'''
See module z for how to initialize un-initialized params.
'''


# x0 = Param(
#     name = 'x0', 
#     mechanism = m_x,
#     init_val = lambda self: 3 * y.y0.val, 
#     update_val = 27)



