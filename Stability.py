# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:01:17 2024

@author: Linne
"""

import numpy as np  
from fixed_constants import *
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_context('paper')

I = np.eye(2,2)
b = b0*2/3
R = gamma*b-c
A = np.array([[-r, -alpha*b],
             [gamma,  R]])
A2 = np.matmul(A,A)
A3 = np.matmul(A2, A)

def P_RK(dt):
    bigterm = 6*I + 3*dt*A + dt**2*A2 +dt**3/4*A3
    P = I + dt/6*np.matmul(A, bigterm)
    return P
def P_heun(dt):
    P = I + dt*A + dt**2/2*A2
    return P
def P_euler(dt):
    P = I+dt*A
    return P
    
    
def stability_plot():
    plt.figure(figsize = (4, 1.7))
    dts = np.linspace(0, 10, 1000)

    schemes = ['RK4', 'Heun', 'Euler']
    schemes_P = [P_RK, P_heun, P_euler]
    for s in range(len(schemes)):
        lambdavals = np.zeros(dts.shape)
        for i in range(len(dts)):
            dt = dts[i]
            eigvals, eigvect = np.linalg.eig(schemes_P[s](dt))
            lambdavals[i] = np.max(np.abs(eigvals))
        plt.plot(dts, lambdavals, label = schemes[s])
        
    plt.hlines(1,0, 10, linestyle = ':', color = 'black', label = r'$\rho = 1$')
    plt.legend()
    plt.ylim(0, 5)
    plt.xlim(0, 10)
    plt.ylabel(r'$\rho $')
    plt.xlabel('$\Delta t$')
