# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:51:44 2024

@author: Linne
"""

from ROMclass import ENSO_ROM
import numpy as np
import matplotlib.pyplot as plt
from AnalysisFuncs import *
import random
import seaborn as sns
from fixed_constants import *
palette = sns.color_palette("gnuplot", 10)[0::4]
sns.set_theme(context='paper', style='white', palette=palette,rc={'xtick.bottom': True,'ytick.left': True, 'figure.dpi' : 300})

mu_crit = 2/3
h_init, T_init = 0, 1.125

def calc_period(T, dt):
    absT = np.abs(T)
    turningpoints = []
    
    #can be made more efficient
    for i in range(len(absT)-1):
        if (absT[i-1] > absT[i]) and (absT[i+1]>absT[i]):
            turningpoints.append(i)
    turningpoints = np.array(turningpoints)
    
    tau = np.mean(turningpoints[1:]- turningpoints[:-1])*2*dt
    std = np.std(turningpoints[1:]- turningpoints[:-1])*2*dt/\
        np.sqrt(len(turningpoints)-1)
    return [tau, std]

def Task_A(endtime = 41, nt = 410):  
    dt = endtime/nt
    sim = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = mu_crit)
    sim.plot_ts_traj()
    tau_c = calc_period(sim.T_ts, dt)
    print(f'The period of oscillation is {np.round(tau_c[0], 4)} +/- \
          {2*dt} months')

def Task_Ba(endtime = 5*41, nt = 5*41):
    sim_bigmu = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = 0.75)
    sim_bigmu.plot_ts_traj()
    
def Task_Bb(endtime = 5*41, nt = 5*41):
    sim_smallmu = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = 0.50)
    sim_smallmu.plot_ts_traj()
    
def Task_Ca(endtime = 5*41, nt = 5*41):
    sim_nl = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = mu_crit, en = 0.1)
    sim_nl.plot_ts_traj()

def Task_Cb(endtime = 5*41, nt = 5*41):
    sim_nl_bigmu = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = 0.75, en = 0.1)
    sim_nl_bigmu.plot_ts_traj()

def Task_D(endtime = 5*41, nt = 5*41):
    sim_varymu = ENSO_ROM(T_init, h_init, 2000, 500, mu_0 = 0.75, en = 0.1, varymu = True)
    sim_varymu.plot_ts_traj()