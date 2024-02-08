# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:51:44 2024

@author: Linne
"""

from ROMclass import ENSO_ROM
from HT_funcs import *
import numpy as np
import matplotlib.pyplot as plt
import random

h_init, T_init = 0, 1.125

def calc_period(T, dt, allT = False, maxtp = np.inf):
    '''
    Function to calculate the period of an oscillation by finding the turning 
    points.

    Parameters
    ----------
    T : numpy array
        Series for which to calculate period.
    dt : float
        interval between entries in T.
    allT : Bool, optional
        Whether to return all individual periods instead of the mean.The 
        default is False.
    maxtp : float, optional
        Maximum value a turning point can have. Needed to calculate period for
        irregular oscillations with multiple minimums/maximums in each period.
        The default is np.inf.

    Returns
    -------
    [period, standard deviation] or a list of all periods (allT = True)

    '''
    absT = np.abs(T)
    turningpoints = []
    #can be made more efficient
    for i in range(len(absT)-1):
        if absT[i]<maxtp:
            if (absT[i-1] > absT[i]) and (absT[i+1]>absT[i]):
                turningpoints.append(i)
    turningpoints = np.array(turningpoints)
    tau = np.mean(turningpoints[1:]- turningpoints[0:-1])*2*dt
    std = np.std(turningpoints[1:]- turningpoints[0:-1])*2*dt/\
        np.sqrt(len(turningpoints)-1)
    if allT:
        return (turningpoints[1:]- turningpoints[:-1])*2*dt
    else:
        return tau, std

def Task_A(endtime = 41, nt = 41*30, pri = False):  
    '''
    Running the linear neutral ROM with mu = mu_c = 2/3 and 
    plotting the time series and trajectories

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 41.
    nt : int, optional
        number of time steps. The default is 41*30.
    pri : Bool, optional
        Whether or not to print period of oscillations and time lag. The 
        default is False.

    Returns
    -------
    None.

    '''
    dt = endtime/nt
    #running the neutral linear ROM
    sim = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = mu_crit)
    #plotting
    sim.plot_ts_traj()
    if pri:
        #calculate the period
        tau_c = calc_period(sim.T_ts, dt)
        tauprint = np.round(tau_c[0], 4)
        print(f'The period of oscillation is {tauprint} +/- {2*dt} months')
        #calculate the time lag
        Ti = np.where(sim.T_ts == sim.T_ts.max())[0][0]
        hi = np.where(sim.h_ts == sim.h_ts.max())[0][0]
        print(f'Time lag between h_w and T is {(hi-Ti)*dt} months')
    

def Task_Ab(endtime = 41*5, nt = 5*41*30):
    '''
    Plotting the variation of the maximum T and h values with the initial T 
    value for the neutral linear ROM

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 41*5.
    nt : int, optional
        number of time steps. The default is 41*30*5.

    Returns
    -------
    None.

    '''
    Ts = np.linspace(0, 5, 100)
    T_amp = np.zeros(100)
    h_amp = T_amp.copy()
    for i in range(100):
        sim = ENSO_ROM(Ts[i], h_init, nt, endtime, mu_0 = mu_crit)
        T_amp[i] = np.max(sim.T_ts)*T_scale
        h_amp[i] = np.max(sim.h_ts)*h_scale/10
    fig, axs = plt.subplots(1, 3, figsize = (10, 2))
    for ax in [axs[0], axs[2]]:
        ax.axis('off')
    ax = axs[1]
    ax.plot(Ts, T_amp, label = '$T_E$')
    ax.plot(Ts, h_amp, label = '$h_w$')
    ax.set_xlabel('$T_{E0}, K$')
    ax.set_ylabel('$T_{Emax} (K), h_{wmax} (10m)$')
    fig.tight_layout()
    res = np.polyfit(Ts, T_amp, 1)
    resh = np.polyfit(Ts, h_amp, 1)
    print(f'The gradient of the T_E and h_w lines are {np.round(res[0], 3)} \
          and {np.round(resh[0], 3)} respectively')

def Task_B(endtime = 5*41, nt = 5*41*30):
    '''
    Running the neutral linear ROM with for mu = 2/3, 0.65 and 0.68, and 
    plotting the time series and trajectories

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 41*5.
    nt : int, optional
        number of time steps. The default is 41*30*5.

    Returns
    -------
    None.

    '''
    colors = ['mediumblue', 'crimson']
    mu_list = [0.65, 0.68]
    sim = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = 2/3)
    fig, axs = sim.plot_ts_traj_sep(returnaxs=True, 
                                    plotlabel = r'$\mu = \frac{2}{3}$',
                                    color = 'black')    
    for i in range(len(mu_list)):
        mu = mu_list[i]
        sim = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = mu)
        sim.plot_ts_traj_sep(returnaxs=False, plotlabel = f'$\mu = {mu}$', 
                             axs=axs, color = colors[i], fig=fig)
    axs[2].legend(bbox_to_anchor=(1., 1.03))

def Task_Bb(endtime = 5*41, nt = 5*41*30):
    '''
    Plotting the variation of period with mu for the neutral linear ROM

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 41*5.
    nt : int, optional
        number of time steps. The default is 41*30*5.

    Returns
    -------
    None.

    '''
    mu_list = np.linspace(0.50, 0.88, 200)
    periods = np.zeros(len(mu_list))
    stds = periods.copy()
    sim = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = 2/3)    
    for i in range(len(mu_list)):
        mu = mu_list[i]
        sim = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = mu_list[i])
        periods[i], stds[i] = calc_period(sim.h_ts, endtime/nt, allT = False)
    fig, axs = plt.subplots(1, 3, figsize = (10, 2))
    for ax in [axs[0], axs[2]]:
        ax.axis('off')
    ax = axs[1]
    ax.plot(mu_list, periods)
    ax.fill_between(mu_list, periods+stds, periods-stds, alpha = 0.4)
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('Period, months')
    fig.tight_layout()

def Task_C(endtime = 5*41, nt = 5*41*30):
    '''
    Running the ROM with non-linearity (en = 0.1) for mu = 2/3, 0.70 and 0.75,
    and plotting the time series and trajectories

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 41*5.
    nt : int, optional
        number of time steps. The default is 41*30*5.

    Returns
    -------
    None.

    '''
    
    sim_nl_c = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = mu_crit, en = 0.1)
    fig, axs = sim_nl_c.plot_ts_traj_sep(plotlabel = '$\mu = \mu_c$', 
                                         returnaxs=True, color = 'black')
    
    sim_nl_1 = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = 0.70, en = 0.1)
    sim_nl_1.plot_ts_traj_sep(axs=axs, fig = fig, plotlabel = '$\mu = 0.70$', 
                              color = 'crimson')

    sim_nl_2 = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = 0.75, en = 0.1)
    sim_nl_2.plot_ts_traj_sep(axs=axs, fig = fig, plotlabel = '$\mu = 0.75$', 
                              color = 'mediumblue')
    
    axs[2].scatter(sim_nl_c.h_ts[-1]*h_scale/10, sim_nl_c.T_ts[-1]*T_scale, 
                   color = 'limegreen', zorder = 5, s = 7, 
                   label = '$\mu = \mu_c$ end' )
    plt.legend(bbox_to_anchor=[1,1])

def Task_D(endtime = 5*41, nt = 5*41*30, pri = False):
    '''
    Running the ROM with non-linearity (en = 0.1) and varying mu for 
    mu_0 = 0.75, and plotting the time series and trajectories

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 41*5.
    nt : int, optional
        number of time steps. The default is 41*30*5.
    pri: bool, optional
        Whether to print period of oscillation. The default is False.

    Returns
    -------
    None.

    '''
    sim_varymu = ENSO_ROM(T_init, h_init, nt, endtime, mu_0 = 0.75, en = 0.1,
                          varymu = True)
    sim_varymu.plot_ts_traj()
    if pri:
        tau_c = calc_period(sim_varymu.T_ts, endtime/nt, maxtp=0.1)
        tauprint = np.round(tau_c[0], 1)
        tauerr = np.round(tau_c[1], 1)
        print(f'The period of oscillation is {tauprint} +/- {tauerr} months')
    
    
def Task_E_fixedmu(endtime = 5*41, nt = 5*41*30):
    '''
    rununing a linear ROM with fixed mu = mu_c and initial values of 0 for T 
    and h for (a) no wind forcing (b) random wind forcing (c) annual wind 
    forcing and (d) random + annual wind forcing

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 41*5.
    nt : int, optional
        number of time steps. The default is 41*30*5.
    pri: bool, optional
        Whether to print period of oscillation. The default is False.

    Returns
    -------
    None.

    '''

    Ws = np.random.random(int(endtime/(1/30))+2)*2-1
    ran= ENSO_ROM(0, 0, nt, endtime, en = 0, varyeta1 = True, mu_0 = mu_crit,
                  f_ann_forcing=False, W_list=Ws)
    fig, axs = ran.plot_ts_traj_sep(plotlabel = 'Random', returnaxs=True, 
                                    color = 'crimson')
    
    forcing= ENSO_ROM(0, 0, nt, endtime, varyeta1 = True, mu_0 = mu_crit,
                      W_list=Ws)
    forcing.plot_ts_traj_sep(axs=axs, fig = fig, plotlabel = 'Annual + Random', 
                             color = 'mediumblue')
    
    sim = ENSO_ROM(0, 0, nt, endtime, mu_0 = mu_crit)
    sim.plot_ts_traj_sep(axs=axs, fig = fig, plotlabel = 'No Forcing', 
                         color = 'black')

    ann= ENSO_ROM(0, 0, nt, endtime, en = 0, varyeta1 = True,
                  mu_0 = mu_crit, f_ran_forcing = False)
    ann.plot_ts_traj_sep(axs=axs, fig = fig, plotlabel = 'Annual', 
                         color = 'green')

    plt.legend(bbox_to_anchor = [1, 1])
    
def Task_E(endtime = 5*41, nt = 5*41*30):
    '''
    rununing a linear ROM with wind forcing and varying mu for various mu_0 
    values (0.65, 2/3, 0.68).

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 41*5.
    nt : int, optional
        number of time steps. The default is 41*30*5.
    Returns
    -------
    None.

    '''
    Ws = np.random.random(int(endtime/(1/30))+2)*2-1
    
    mu1= ENSO_ROM(T_init, h_init, nt, endtime, varyeta1 = True, 
                  varymu = True, mu_0 = 0.68, W_list = Ws)
    fig, axs = mu1.plot_ts_traj_sep(plotlabel = '$\mu_0 = 0.68$', 
                                    returnaxs=True, color = 'crimson')
    
    mu2= ENSO_ROM(T_init, h_init, nt, endtime, varyeta1 = True, varymu = True,
                  mu_0 = mu_crit, W_list=Ws)
    mu2.plot_ts_traj_sep(axs=axs, fig = fig, plotlabel = '$\mu_0 = \mu_c$',
                         color = 'black')
    
    mu3= ENSO_ROM(T_init, h_init, nt, endtime, varyeta1 = True, varymu = True,
                  mu_0 = 0.65, W_list=Ws)
    mu3.plot_ts_traj_sep(axs=axs, fig = fig, plotlabel = '$\mu_0 = 0.65$',
                         color = 'mediumblue')
    plt.legend(bbox_to_anchor = [1, 1])
    
def Task_E_varydt(endtime = 5*41):
    '''
    rununing a linear ROM with varying mu, where mu_0 = mu_c, and wind stress 
    forcing for various timestep size (0.5, 1, 5 days). Random values used 
    across models are kept constant so that comparisons can be made.

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 5*41.

    Returns
    -------
    None.

    '''
    Ws = np.random.random(int(endtime/(1/30))+2)*2-1
    
    dt1= ENSO_ROM(T_init, h_init, int(endtime *30/5), endtime, 
                  varyeta1 = True, varymu = True, mu_0 = mu_crit, W_list = Ws)
    fig, axs = dt1.plot_ts_traj_sep(plotlabel = '$\\Delta t = 5 days$', 
                                    returnaxs=True, color = 'mediumblue')
    
    dt2= ENSO_ROM(T_init, h_init, endtime * 30, endtime, en = 0, 
                  varyeta1 = True, varymu = True, mu_0 = mu_crit, W_list=Ws)
    dt2.plot_ts_traj_sep(axs=axs, fig = fig, plotlabel = '$\Delta t = 1 days$',
                         color = 'black')
    
    dt3= ENSO_ROM(T_init, h_init, int(endtime *30/0.5), endtime , en = 0, 
                  varyeta1 = True, varymu = True, mu_0 = mu_crit, W_list=Ws)
    dt3.plot_ts_traj_sep(axs=axs, fig = fig, plotlabel='$\Delta t = 0.5 days$',
                         color = 'crimson')
    plt.legend(bbox_to_anchor = [1, 1])

def Task_F(endtime = 5*41, nt = 5*41*30, mu_0 = 0.75, heating = False):
    '''
    Running a non-linear (en = 0.1) model with varying mu and wind stress 
    forcing. mu_0 = 0.75.

    Parameters
    ----------
    endtime : float, optional
        Total time for which to run . The default is 41*5.
    nt : int, optional
        number of time steps. The default is 41*30*5.
    mu_0 : float, optional
        value of mu_0 to use. The default is 0.75.
    heating : float, optional
        Magnitude of random heating. The default is 0.
    Returns
    -------
    None.

    '''
    sim= ENSO_ROM(T_init, h_init, nt, endtime, en = 0.1, varyeta1 = True, 
                  varymu = True, mu_0 = mu_0, varyeta2 = heating)
    sim.plot_ts_traj()
    
def Task_G(Trange = 0.5, hrange = 5, ensemble_size = 50, endtime = 60*12,
           heating = False):
    '''
    Runs an ensemble of 50 models with non-linearity (en = 0.1), time varying 
    mu (mu_0 = 0.75) and wind stress forcing. The initial conditions are 
    randomly perturbed such that T_init in [T-Trange, T+T_range] and h_init in
    [h-hrange, h+h_range]. Plume diagrams are plotted for both T and h. 

    Parameters
    ----------
    Trange : float, optional
        Maximum deviation of T_init from 1.125. The default is 0.5.
    hrange : float, optional
        Maximum deviation of h_init from 0. The default is 5.
    ensemble_size : int, optional
        number of ensemble members. The default is 50.
    endtime : float, optional
        total time to run the models for. The default is 60*12 (60 years)
    heating : float, optional
        Magnitude of random heating. The default is 0.

    Returns
    -------
    None.

    '''
    nt = int(endtime/day_dim)    
    fig, axs = plt.subplots(2, 3, sharex = True, figsize = (16,3),
                            width_ratios=[1, 4, 1])
    timearr = np.arange(nt+1)*day_dim
    for i in range(ensemble_size):
        T = T_init + random.uniform(-1,1)*Trange
        h = h_init+ random.uniform(-1,1)*hrange
        sim = ENSO_ROM(T, h, nt, endtime, en = 0.1, varyeta1 = True, 
                       varymu = True, mu_0 = 0.75, varyeta2 = heating)
        a = axs[0, 1].plot(timearr, sim.T_ts*T_scale, color = 'mediumblue',
                           alpha = 0.1)
        b = axs[1, 1].plot(timearr, sim.h_ts*h_scale/10, color = 'crimson',
                           alpha = 0.1)
    axs[0,1].set_ylabel('$T_E, ^\circ C$')
    axs[1,1].set_ylabel('$h_w, 10m$')
    axs[1, 1].set_xlabel('Time, months')
    for ax in [axs[0, 0], axs[1, 0], axs[0, 2], axs[1, 2]]:
        ax.axis('off')