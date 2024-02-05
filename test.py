# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:26:42 2024

@author: Linne
"""

import numpy as np
import matplotlib.pyplot as plt
from HT_funcs import *

#probably put these in a different module? to avoid names overlapping
mu = 2/3
h_init = 0
T_init = 1.125
endtime = 41
nt = 1000
dt = endtime/nt



b = b0*mu
R = gamma*b-c

en = 0
eta1 = 0
eta2 = 0


h_ts = np.zeros(nt+1)
T_ts = h_ts.copy()
i = 0
h_ts[0] = h_init
T_ts[0] = T_init

#move in time
#first time step using FT?
    
for i in range(1, nt+1):
    #forward time
    # h_ts[i] = hw_FT(h_ts[i-1], T_ts[i-1], dt, r, alpha, b, eta1)
    # T_ts[i] = TE_FT(h_ts[i], T_ts[i-1], dt, R, gamma, en, b, eta1, eta2)
    
    #4th order runge kutta
    Ti, hi = RK4(h_ts[i-1], T_ts[i-1], dt, r, alpha, b, R, gamma, en, eta1, eta2)
    h_ts[i] = hi
    T_ts[i] = Ti

fig, ax0 = plt.subplots()
p1, = ax0.plot(np.arange(nt+1),h_ts, label = r'$h_w$', linestyle = '-', color = 'b')
ax0.set_ylabel(r'$h_w, 10m$', color = 'b')
ax1 = ax0.twinx()
p2, = ax1.plot(np.arange(nt+1),T_ts, label = r'$T_E$', linestyle = '--', color = 'black')
ax1.set_ylabel('$T_E, ^\circ C$', color = 'black')
ax1.legend(handles=[p1, p2])


plt.figure()
plt.plot(T_ts, h_ts/10)
plt.ylabel('$T_E, ^\circ C$')
plt.xlabel(r'$h_w, 10m$')
