# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:08:51 2024

@author: Linne
"""
import numpy as np
import matplotlib.pyplot as plt
from HT_funcs import *
from fixed_constants import *
import seaborn as sns
palette = sns.color_palette("gnuplot_r", 10)

class ENSO_ROM():
    def __init__(self, T_init, h_init, nt, endTime, en = 0,
                 varyeta1 = False, varyeta2 = False, varymu = False, mu_0 = 2/3, f_ran_forcing = True):
        #non-dimensionalising variables
        self.T = T_init/T_scale
        self.h = h_init/h_scale
        self.endTime = endTime/t_scale        
        
        #calculating time-step size based on endtime and number of time steps
        self.dt = self.endTime/nt
        
        #creating lists to store h and T values
        self.h_ts = np.zeros(nt + 1)
        self.T_ts = self.h_ts.copy()
        #setting variables to keep track of nt and time
        self.nt = 0
        self.time = 0
        self.nt_max = nt
        
        #setting up initial conditions
        self.h_ts[0] = self.h
        self.T_ts[0]= self.T

        #setting up variables that are kept constant throughout
        self.en = en
        self.mu_0 = mu_0
        
        #storing choice to vary mu or use a constant value
        self.varymu = varymu
        #setting up first value of mu
        
        #setting up constant value of mu
        self.b = b0*mu_0
        
        self.R = gamma*self.b-c
        
        #storing choice to vary eta1 or use eta1 = 0
        self.varyeta1 = varyeta1
        self.f_ran_forcing = f_ran_forcing
        self.eta1 = 0
        
        self.varyeta2 = varyeta2
        
        #propogating by nt_max timesteps
        for i in range(self.nt_max):
            self.one_timestep()
        
    def hT_update_func(self):
        #calculating next h value here using RK4
        self.T, self.h = RK4(self.h, self.T, self.dt, self.time, self.mu_0, r,
                             alpha, self.b, self.R, gamma, self.en, self.eta1,
                             varyeta1=self.varyeta1, varymu=self.varymu, 
                             f_ran_forcing=self.f_ran_forcing, 
                             varyeta2 = self.varyeta2)
    
    def update_values(self):
        #add new h and T values to list
        self.h_ts[self.nt] = self.h
        self.T_ts[self.nt]= self.T
        
    def one_timestep(self):
        #move the system by dt
        self.hT_update_func()
        
        #update everything
        self.nt += 1
        self.time += self.dt
        self.update_values()


    def plot_timeseries_sep(self, save = False):
        fig, ax = plt.subplots(2, sharex = True)
        p0, = ax[0].plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.h_ts*h_scale/10, label = r'$h_w$')
        ax[0].set_ylabel(r'$h_w, 10m$')
        p1, = ax[1].plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.T_ts*T_scale, label = r'$T_E$'
                        , linestyle = '--')
        ax[1].set_ylabel('$T_E, ^\circ C$', color = 'black')
        ax[1].set_xlabel('Time, months')
                
        if save:
            fig.savefig(save, bbox_inches='tight')
            
    def plot_timeseries_tgt(self, save = False):
        fig, ax = plt.subplots()
        p0, = ax.plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.h_ts*h_scale/10, label = r'$h_w$')
        ax.set_ylabel(r'$h_w, 10m$')
        ax.set_xlabel('Time, months')
        ax.set_ylim(-5, 5)
        
        # ax1 = ax.twinx()
        p1, = ax.plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.T_ts*T_scale, label = r'$T_E$'
                        , linestyle = '--', color = palette[1])
        # ax1.set_ylabel('$T_E, ^\circ C$', color = palette[1])
        ax.legend(handles=[p0, p1])
        if save:
            fig.savefig(save, bbox_inches='tight')


        
    def plot_trajectory(self, save = False):
        plt.figure()
        plt.plot(self.h_ts*h_scale/10, self.T_ts*T_scale, 
                 label = 'Trajectory')  
        plt.scatter(self.h_ts[0]*h_scale/10, self.T_ts[0]*T_scale,
                    label = 'Start', marker = '.', color = 'red', zorder = 5)
        plt.ylabel('$T_E, ^\circ C$')
        plt.xlabel(r'$h_w, 10m$')
        plt.legend()
        if save:
            plt.savefig(save, bbox_inches='tight')
    
    def plot_ts_traj(self):
        fig, axs = plt.subplots(1,2, width_ratios = (2,1), figsize = (10, 3))
        p0, = axs[0].plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.h_ts*h_scale/10, label = r'$h_w$', color = palette[8])
        axs[0].set_ylabel(r'$h_w \ (10m), T_E \ (^\circ C)$')
        axs[0].set_xlabel('Time, months')
        
        p1, = axs[0].plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.T_ts*T_scale, label = r'$T_E$'
                        , linestyle = '-', color = palette[4])
        axs[0].legend(handles=[p0, p1])
        axs[0].set_title('(a) Time Series of $h_w$ and $T_E$', y = -0.35)
        
        axs[1].plot(self.h_ts*h_scale/10, self.T_ts*T_scale, 
                 label = 'Trajectory') 
        axs[1].scatter(self.h_ts[0]*h_scale/10, self.T_ts[0]*T_scale,
                    label = 'Start', marker = '.', color = 'red', zorder = 5)
        axs[1].set_xlabel('$T_E, ^\circ C$')
        axs[1].set_ylabel(r'$h_w, 10m$')
        axs[1].legend()
        axs[1].set_title('(b) Trajectory Plot', y = -0.35)
        fig.tight_layout()

    
        
    