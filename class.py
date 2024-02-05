# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:08:51 2024

@author: Linne
"""
import numpy as np
import matplotlib.pyplot as plt
from HT_funcs import *
from fixed_constants import *

class ENSO_ROM():
    def __init__(self, T_init, h_init, dt, endTime, mu = 2/3, en = 0,
                 eta1 = 0, eta2 = 0):
        self.dt = dt
        self.h_ts = np.zeros(endTime/dt + 1)
        self.T_ts = self.h_ts.copy()
        self.nt_max = endTime/dt
        self.nt = 0
        self.T = T_init
        self.h = h_init
        
        self.h_ts[0] = h_init
        self.T_ts[0]= T_init

        
        self.b = b0*mu
        self.R = gamma*self.b-c
        
    def hT_update_func(self):
        #calculating next h value here
        self.T, self.H = RK4(self.h, self.T, dt, r, alpha, self.b, self.R, 
                             gamma, en, eta1, eta2)
    
    def update_timestep(self):
        self.nt += 1
        self.h_ts[nt] = self.h
        self.T_ts[nt]= self.T
        
    def one_timestep(self):
        self.hT_update_func()
        self.update_timestep()

        
        
    def plot_timeseries(self):
        fig, ax0 = plt.subplots()
        p0, = ax0.plot(np.arange(len(self.h_ts)),self.h_ts, label = r'$h_w$')
        ax0.set_ylabel(r'$h_w, 10m$', color = 'b')
        ax1 = ax0.twinx()
        p1, = ax1.plot(np.arange(len(self.h_ts)),self.T_ts, label = r'$T_E$')
        ax1.set_ylabel('$T_E, ^\circ C$', color = 'black')
        
        ax1.legend(handles=[p0, p1])
    def plot_trajectory(self):
        plt.figure()
        plt.plot(self.T_ts, self.h_ts/10)
        plt.ylabel('$T_E, ^\circ C$')
        plt.xlabel(r'$h_w, 10m$')

        
    
        
    