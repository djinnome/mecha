#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:18:59 2022

@author: prcohen
"""

import os, time
os.chdir("/Users/zuck016/Projects/Proposals/ASKEM")

from mecha.classes.registry import Registry
from mecha.classes.param_classes import Param
from mecha.classes.mechanism_classes import Mechanism

m_y = Mechanism(name = 'm_y', n = 20)
current_mechanism = m_y


'''
In this case y0 (which, recall, is required by x0) has an update_val that
sets its value to the value of x0.  Note that we don't say init_now = False 
for y0 because init_val contains no foreign params, so y0 can be initialized right away

'''

y0 = Param(
    name = 'y0', 
    init_val = 51, 
    update_val = lambda self: Registry.get('m_x','x0').val
    )



# y0 = Param(
#     name = 'y0', 
#     mechanism = m_y,
#     init_val = 51, 
#     update_val = lambda self: x.x0.val
#     )
