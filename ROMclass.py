# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:08:51 2024

@author: Linne
"""
import numpy as np
import matplotlib.pyplot as plt
from HT_funcs import *
import seaborn as sns
import matplotlib.gridspec as gridspec
sns.set_theme(context='paper', style='whitegrid',rc={'xtick.bottom': True,'ytick.left': True, 'figure.dpi' : 300})

class ENSO_ROM():
    def __init__(self, T_init, h_init, nt, endTime, en = 0, varyeta1 = False, 
                 varyeta2 = False, varymu = False, mu_0 = 2/3, 
                 f_ran_forcing = True, f_ann_forcing = True, W_list = []):
        '''
        Initialise and run the ROM model of coupled h, thermocline depth, and
        T, temperature anomaly. The model uses 4th order Runge Kutta.

        Parameters
        ----------
        T_init : float
            Initial T value in K.
        h_init : float
            initial h value in m.
        nt : int
            number of time steps.
        endTime : float
            total time to run model for in months.
        en : float, optional
            degree of non-linearity. The default is 0. A typical value is 0.1
        varyeta1 : Bool, optional
            Whether to include noisy wind forcing. The default is False.
        varyeta2 : float, optional
            Magnitude of random heating. The default is 0.
        varymu : Bool, optional
            Whether to vary mu on an annual cycle. The default is False.
        mu_0 : float, optional
            Controls the coupling coefficient
            mu if varymu = False, mu_0 if varymu = True. The default is 2/3.
        f_ran_forcing : Bool, optional
            Whether to include random component of forcing if varyeta1 = True.
            The default is True.
        f_ann_forcing : Bool, optional
            Whether to include annual component of forcing if varyeta1 = True.
            The default is True.
        W_list : Array, optional
            List of random values between -1 and 1, of length at least 
            endtime/tau_corr. For using same W values across different models.
            The default is [].

        Returns
        -------
        None.

        '''
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
        self.f_ann_forcing = f_ann_forcing
        self.eta1 = 0
        if len(W_list)==0:
            self.W = np.random.random(int(self.endTime/tau_cor)+2)*2-1
        else:
            self.W = W_list
        
        self.varyeta2 = varyeta2
        
        #propogating by nt_max timesteps
        for i in range(self.nt_max):
            self.one_timestep()
        
    def hT_update_func(self):
        '''
        Calculating and updating values of T and h using RungeKutta 4.
        '''
        #calculating next h value here using RK4
        self.T, self.h = RK4(self.h, self.T, self.dt, self.time, self.mu_0, r,
                             alpha, self.b, self.R, gamma, self.en, self.eta1,
                             varyeta1=self.varyeta1, varymu=self.varymu, 
                             f_ran_forcing=self.f_ran_forcing, 
                             f_ann_forcing=self.f_ann_forcing,
                             varyeta2 = self.varyeta2, W_list = self.W)
    
    def update_values(self):
        '''
        Updating h and T time series with current values of h and T

        '''
        self.h_ts[self.nt] = self.h
        self.T_ts[self.nt]= self.T
        
    def one_timestep(self):
        '''
        Moves the system by 1 timestep (dt) and updates all time series and 
        values

        '''
        self.hT_update_func()
        
        #update everything
        self.nt += 1
        self.time += self.dt
        self.update_values()

    
    def plot_ts_traj(self):
        '''
        Plot T and h time series on the same subplot, and plots the trajectory 
        next to it

        '''
        fig, axs = plt.subplots(1,2, width_ratios = (2,1), figsize = (10, 3))
        p0, = axs[0].plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.h_ts*h_scale/10, label = r'$h_w$', color = 'crimson')
        axs[0].set_ylabel(r'$h_w \ (10m), T_E \ (K)$')
        axs[0].set_xlabel('Time, months')
        
        p1, = axs[0].plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.T_ts*T_scale, label = r'$T_E$'
                        , linestyle = '-', color = 'mediumblue')
        axs[0].legend(handles=[p0, p1])
        axs[0].set_title('(a) Time Series of $h_w$ and $T_E$', y = -0.35)
        
        axs[1].plot(self.h_ts*h_scale/10, self.T_ts*T_scale, 
                 label = 'Trajectory', color = 'mediumblue') 
        axs[1].scatter(self.h_ts[0]*h_scale/10, self.T_ts[0]*T_scale,
                    label = 'Start', marker = '.', color = 'red', zorder = 5)
        axs[1].set_xlabel('$T_E, K$')
        axs[1].set_ylabel(r'$h_w, 10m$')
        axs[1].legend()
        axs[1].set_title('(b) Trajectory', y = -0.35)
        fig.tight_layout()
        
    def plot_ts_traj_sep(self, axs = False, color = 'b', returnaxs = False,
                         plotlabel = None, fig = None):
        '''
        Plots the time series of T, h and the trajectory on separate sub plots 
        in the same figure

        Parameters
        ----------
        axs : list of ax objects, optional
            [axh, axT, axTraj] if you want to plot the plots on existing axes.
            The default is False.
        color : string, optional
            color to plot the timeseries and trajectory in. The default is 'b'.
        returnaxs : Bool, optional
            Whether to return the list of axes and the figure object. The 
            default is False.
        plotlabel : String, optional
            Label to put in the legend. The default is None.
        fig : matplotlib figure object, optional
            figure object to plot on. The default is None.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        list
            DESCRIPTION.

        '''
        if axs == False:
            fig = plt.figure(figsize = (10, 3), layout = 'constrained')
            spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios = (5,2))
            axh = fig.add_subplot(spec[0, 0])
            axT = fig.add_subplot(spec[1, 0], sharex=axh)
            plt.setp(axh.get_xticklabels(), visible=False)
            axTraj = fig.add_subplot(spec[:, 1])
            axTraj.scatter(self.h_ts[0]*h_scale/10, self.T_ts[0]*T_scale,
                        label = 'Start', marker = '.', color = 'red', zorder = 15)

        else:
            axh = axs[0]
            axT = axs[1]
            axTraj = axs[2]
            
        p0, = axh.plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.h_ts*h_scale/10, label = r'$h_w$', color = color)
        axh.set_ylabel(r'$h_w \ (10m)$')
        axT.set_xlabel('Time, months')
        axT.set_title('(a) Time series of $h_w$ and $T_E$', y = -0.75)
        
        p1, = axT.plot(np.arange(len(self.h_ts))*self.dt*t_scale,
                        self.T_ts*T_scale, linestyle = '-', color = color)
        axT.set_ylabel('$T_E, K$')        
        axTraj.plot(self.h_ts*h_scale/10, self.T_ts*T_scale, color = color,
                    label = plotlabel, alpha = 0.6) 
        axTraj.set_xlabel('$T_E, K$')
        axTraj.set_ylabel(r'$h_w, 10m$')
        axTraj.set_title('(b) Trajectory', y = -0.34)

        if returnaxs:
            return fig, [axh, axT, axTraj]


    
        
    