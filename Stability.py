# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:01:17 2024

@author: Linne
"""

import numpy as np  
from HT_funcs import *
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_context('paper')

I = np.eye(2,2) #identity matrix
b = b0*mu_crit
R = gamma*b-c
#d/dt (h, T) = A x (h,t)
A = np.array([[-r, -alpha*b],
             [gamma,  R]])
A2 = np.matmul(A,A)
A3 = np.matmul(A2, A)

def P_RK(dt):
    '''
    P for runge kutta 4 scheme

    Parameters
    ----------
    dt : float
        timestep.

    Returns
    -------
    P : 2x2 array

    '''
    bigterm = 6*I + 3*dt*A + dt**2*A2 +dt**3/4*A3
    P = I + dt/6*np.matmul(A, bigterm)
    return P
def P_heun(dt):
    '''
    P for heun scheme

    Parameters
    ----------
    dt : float
        timestep.

    Returns
    -------
    P : 2x2 array

    '''
    P = I + dt*A + dt**2/2*A2
    return P
def P_euler(dt):
    '''
    P for euler scheme

    Parameters
    ----------
    dt : float
        timestep.

    Returns
    -------
    P : 2x2 array

    '''
    P = I+dt*A
    return P
    
    
def stability_plot():
    '''
    Plots spectral radius for dt between 0 and 10 for RK4, heun and euler.

    Returns
    -------
    None.

    '''
    fig, axs = plt.subplots(1, 3, figsize = (10, 2))
    dts = np.linspace(0, 10, 1000)
    for ax in [axs[0], axs[2]]:
        ax.axis('off')
    colors = ['black', 'mediumblue', 'crimson']
    schemes = ['RK4', 'Heun', 'Euler']
    schemes_P = [P_RK, P_heun, P_euler]
    for s in range(len(schemes)):
        lambdavals = np.zeros(dts.shape)
        for i in range(len(dts)):
            dt = dts[i]
            eigvals, eigvect = np.linalg.eig(schemes_P[s](dt))
            lambdavals[i] = np.max(np.abs(eigvals))
        axs[1].plot(dts, lambdavals, label = schemes[s], color = colors[s])
        
    axs[1].hlines(1,0, 10, linestyle = ':', color = 'black', label = r'$\rho = 1$')
    axs[1].legend()
    axs[1].set_ylim(0, 5)
    axs[1].set_xlim(0, 10)
    axs[1].set_ylabel(r'$\rho $')
    axs[1].set_xlabel('$\Delta t$')
    fig.tight_layout()

