#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 21:57:32 2022

@author: prcohen
"""

import inspect

class Registry ():
    
    mechanisms = {}
    foreigns = []
    
    def register (mechanism):
        ''' Registers mechanisms.  Within a mechanism all the objects are stored
        in a dict called objects.'''
        r = Registry.mechanisms
        n = mechanism.name
        
        if r.get(n) is None:
            r.update({n:mechanism})
        else:
            print(f"\nWARNING: Overwriting mechanism {n} in Registry.")
            r[n] = mechanism
    
    def get (mech_name,obj_name):
        try: 
            return Registry.mechanisms[mech_name].objects[obj_name]
        except KeyError as e:
            if Registry.mechanisms.get(mech_name) is not None:
                raise KeyError(f"Registry mechanism {mech_name} doesn't contain a object called {e}")
                
            else:
                raise KeyError(f"Registry doesn't contain a mechanism called {e}")
                
    def initialize_all_mechanisms ():
        for i in range(2):
            for mech in Registry.mechanisms.values():
                try:
                    mech.initialize()
                except:
                    pass
                
    
        
        
                
            
                

# class A ():
#     def __init__(self,name):
#         self.name = name
#         self.mechanism = 'A'

# a1 = A('foo')
# a2 = A('bar')

# class B ():
#     def __init__(self,name):
#         self.mechanism = 'B'
#         self.name = name


# b1 = B('baz')
# b2 = B('bik')

# for obj in [a1,b1,a2,b2]:
#     Registry.register(obj.mechanism,obj.name,obj)
#     print("\n",Registry.objects)
    

# print(Registry.get('A','foo'))