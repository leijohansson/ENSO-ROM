# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:01:17 2024

@author: Linne
"""

import numpy as np  
from fixed_constants import *
import seaborn as sns
sns.set_style('white')
def stability_plot():
    plt.figure(figsize = (4, 1.5))
    b = b0*2/3
    R = gamma*b-c
    A = np.array([[-r, -alpha*b],
                 [gamma,  R]])
    
    I = np.eye(2,2)
    
    dts = np.linspace(0, 10, 1000)
    RK4  = np.zeros(dts.shape)
    A2 = np.matmul(A,A)
    A3 = np.matmul(A2, A)
    
    for i in range(len(dts)):
        dt = dts[i]
        bigterm = 6*I + 3*dt*A + dt**2*A2 +dt**3/4*A3
        P = I + dt/6*np.matmul(A, bigterm)
        eigvals, eigvect = np.linalg.eig(P)
        RK4[i] = np.max(np.abs(eigvals))
    
    plt.plot(dts, RK4, label = 'RK4')
    
    euler  = np.zeros(dts.shape)
    for i in range(len(dts)):
        dt = dts[i]
        P = I+dt*A
        eigvals, eigvect = np.linalg.eig(P)
        euler[i] = np.max(np.abs(eigvals))
    plt.plot(dts, euler, label = 'Euler')
    
    heun = np.zeros(dts.shape)
    for i in range(len(dts)):
        dt = dts[i]
        P = I + dt*A + dt**2/2*A2
        eigvals, eigvect = np.linalg.eig(P)
        heun[i] = np.max(np.abs(eigvals))
    plt.plot(dts, heun, label = 'Heun')
    plt.hlines(1,0, 10, linestyle = ':', color = 'black', label = '$|\lambda| = 1$')
    plt.legend()
    plt.ylim(0, 5)
    plt.xlim(0, 10)
    plt.ylabel('$|\lambda|$')
    plt.xlabel('$\Delta t$')
stability_plot()