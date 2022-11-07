#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 20:34:14 2022

@author: prcohen
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import ternary

import numpy as np
import pandas as pd



    

def plot_series (*y,x=None,label=None,color=None,xlim = None, ylim=None, xlabel= None, ylabel=None,title = None, figsize=(8,6)):
    ''' xlim and ylim must be lists of the form [lower_bound,upper_bound]'''
    fig, ax = plt.subplots(figsize=figsize)
    ax = plt.gca()
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    for _y in y:
        if x is None:
            ax.plot(np.arange(len(_y)), _y, label=label,color=color,linewidth=.5)
        else:
            ax.plot(x, _y, label=label,color=color,linewidth=.5)
    if title is not None: plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(xlabel)   



def plot_multiseries (df,x=None,standardize=False, xlabel = None, ylabel = None, title = None, figsize = None):
    ''' Be sure that the first column in df is the x index, or specify which column is x '''
    fig, ax = plt.subplots(figsize= figsize or (8,6))
    if x is None: x = df.columns[0]
    
    if standardize:
        _df = pd.DataFrame(df[x])
        m = df.mean()
        s = df.std()
        for col in df.columns:
            if not col == x:
                _df['Z_'+col] = (df[col] - m[col])/s[col]
    else:
        _df = df
                   
    for col in _df.columns:
        if not col == x:
            plt.plot(_df[x], _df[col], label='Line '+col,linewidth=1)

    if title is not None : plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)

    #add legend
    plt.legend()

    #display plot
    plt.show() 
    
    
def plot_replicates (df,x=None,standardize=False, figsize=(8,6), xlabel = None, ylabel = None, xlim=None, ylim=None, title = None, legend=True,color = True,filepath=None):
    ''' This expects a pandas df in which x is the values on the x axis, the 'trial' 
    column identifies the replicate and the rest of the columns are y variates. 
    xlim and ylim each have the form [lower,upper]'''
    fig, ax = plt.subplots(figsize=figsize)
    if x is None: x = df.columns[0]
    
    if 'trial' not in df.columns:
        raise ValueError("To plot replicates, df must have a column named 'trial'. ")
    for trial in df.trial.unique():
        df_trial = df[df['trial']==trial]
        
        if standardize:
            _df = pd.DataFrame(df_trial[x])
            m = df_trial.mean()
            s = df_trial.std()
            for col in df_trial.columns:
                if not col == x and not col == 'trial':
                    _df['Z_'+col] = (df_trial[col] - m[col])/s[col]
        else:
            _df = df_trial
                   
        for col in _df.columns:
            if not col == x and not col == 'trial':
                if color == True:
                    plt.plot(_df[x], _df[col], label=str(trial)+'_'+col,linewidth=1)
                else:
                    plt.plot(_df[x], _df[col], label=str(trial)+'_'+col,linewidth=1,c='black')
                    
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
        
    if title is not None : plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)

    #add legend
    if legend: plt.legend()

    #display or save plot
    if filepath is not None: 
        plt.savefig(filepath)  
    
    plt.show()       



def plot_scatter (x,y, s=10, c = '0', xlabel = 'X', ylabel = 'Y', title=""):
    plt.scatter(x, y, s, color=c, alpha=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
def multi_hist (*distributions, title = "", subtitles = None, 
                sharex = True, sharey=True, tight_layout=True, 
                figsize = (9,4),bins=20,xlim=None,ylim=None,
                xlabel = None):
    k = len(distributions)
    fig, axs = plt.subplots(1, k, sharey=sharey, sharex=sharex, tight_layout=tight_layout,figsize = figsize)
    
    for d,i in zip(distributions,np.arange(k)): 
        axs[i].hist(d, bins=bins,)
        if subtitles is not None:
            axs[i].set_title(subtitles[i])
        if xlim is not None:
            axs[i].set_xlim(xlim)
        if ylim is not None:
            axs[i].set_ylim(ylim)
        if xlabel is not None:
            axs[i].set_xlabel(xlabel)
            
    plt.suptitle(title, y=1.05, size=16)
    plt.show()

        

def plot_ternary (vertex_labels, points, source_points = None, special_points = None, color_by=None, 
                  color_by_label=None, bounds = None, figsize = (4,4)):
    
    ''' wraps Marc Harper's python-ternary package https://github.com/marcharper/python-ternary'''
    
    mpl.rcParams['figure.dpi'] = 200
    mpl.rcParams['figure.figsize'] = figsize
    
    
    ### Scatter Plot
    scale = 1.0
    fontsize = 2*figsize[0]
    offset = 0.03
    figure, tax = ternary.figure(scale=scale)
    #tax.set_title("Decision Space", fontsize=12)
    tax.boundary(linewidth= .5)
    
    # tax.left_corner_label(vertex_labels[0], fontsize= (3*figsize[0]))
    # tax.top_corner_label(vertex_labels[1], fontsize= (3*figsize[0]))
    # tax.right_corner_label(vertex_labels[2], fontsize= (3*figsize[0]))
    
    tax.left_axis_label(f"<-- Prob. of {vertex_labels[2]}", fontsize=fontsize, offset = .2)
    tax.right_axis_label(f"<-- Prob. of {vertex_labels[1]}", fontsize=fontsize, offset = .2)
    tax.bottom_axis_label(f"Prob. of {vertex_labels[0]} -->", fontsize=fontsize, offset = .2)
    tax.get_axes().axis('off')
    
    tax.gridlines(multiple=0.2, color="black")
    tax.clear_matplotlib_ticks()
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, fontsize=fontsize, offset=offset, tick_formats="%.1f")
    
    
    if color_by is not None:
        cmap = plt.cm.RdYlGn
        if bounds is not None:
            norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
        else:
            norm = mpl.colors.Normalize(vmin=np.min(color_by), vmax=np.max(color_by))
        color = cmap(norm(list(color_by)))
        
        figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            orientation='horizontal', label=color_by_label)
        tax.scatter(points, marker='o', color=color, s = 1, cmap=cmap)
    else:
        tax.scatter(points, marker='o', color='black', s = 1)
    
    # We may want to plot the vectors from source_points to points
    # if so, source_points must have the same shape as points
    if source_points is not None:
        for row in range(len(points)):
            tax.line(source_points[row],points[row], linewidth=.5, color='green')
        
    
    if special_points is not None:
        for p,c in zip(special_points[0],special_points[1]):       
            tax.scatter([p], marker='s', color = [c], s = 10)
        
    tax.gridlines(multiple=5, color="blue")
    #tax.legend(loc = 'upper right',cmap=cmap)
    #tax.ticks(axis='lbr', linewidth=1, multiple=5)
    
    
    tax.show()
    ternary.plt.show()





