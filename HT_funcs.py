# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:18:04 2024

@author: Linne
"""

import numpy as np

#variables that are fixed throughout all models/parts
b0 = 2.5
gamma = 0.75
c=1
r = 0.25
alpha = 0.125
T_scale = 7.5
h_scale = 150
t_scale = 2
mu_crit = 2/3

tau = 12/t_scale #one year
f_ann = 0.02
f_ran = 0.2
day_dim = 1/30
mu_ann = 0.2
tau_cor = 1/30/t_scale #one day

def f_hw(h_w, T_E, r, alpha, b, eta1):
    '''
    Calculating dh/dt

    Parameters
    ----------
    h_w : float
        nondimensionalised value of h (thermocline depth) at that time step. 
    T_E : float
        nondimensionalised value of T (temperature) at that time step.
    r : float
        damping of upper ocean heat content.
    alpha : float
        relates enhanced easterly wind stress to recharge of ocean heat content
    b : float
        measure of thermocline slope. b = 2.5*mu, the coupling coefficient
    eta1 : float
        Wind stress forcing.

    Returns
    -------
    f : float
        dh/dt calculated using the above variables.

    '''
    f = -r*h_w - alpha*b*T_E - alpha*eta1
    return f

def f_TE(h_w, T_E, R, gamma, en, b, eta1, varyeta2 = False): 
    '''
    Calculating dT/dt

    Parameters
    ----------
    h_w : float
        nondimensionalised value of h (thermocline depth) at that time step.
    T_E : float
        nondimensionalised value of T (temperature) at that time step.
    R : float
        describes Bjerknes positive feedback process.
    gamma : float
        feedback of the thermocline gradient on the SST.
    en : float
        controls the degree of non-linearity.
    b : float
        measure of thermocline slope. b = 2.5*mu, the coupling coefficient
    eta1 : float
        Wind stress forcing.
    varyeta2 : Bool, optional
        Whether to add random heating. The default is False.

    Returns
    -------
    Returns
    -------
    f : float
        dT/dt calculated using the above variables.

    '''
    eta2 = 0
    if varyeta2>0:
        eta2 = varyeta2*(np.random.random()*2-1)
    f = R*T_E + gamma*h_w - en*(h_w + b*T_E)**3 + gamma*eta1 + eta2
    return f
    
def RK4(h, T, dt, time, mu_0, r, alpha, b, R, gamma, en, eta1,varyeta1 = False,
        varymu = False, f_ran_forcing=True, f_ann_forcing = True, 
        varyeta2  = False, W_list = None):
    '''
    

    Parameters
    ----------
    h : float
        nondimensionalised value of h (thermocline depth) at current time step.
    T : float
        nondimensionalised value of T (temperature anomaly) at the current time step.
    dt : float
        time step, nondimensionalised.
    time : float
        current time, nondimensionalised.
    mu_0 : float
        coupling coefficient if varymu = False, controls coupling coefficient
        is varymu = True.
    r : float
        damping of upper ocean heat content.
    alpha : float
        relates enhanced easterly wind stress to recharge of ocean heat content
    b : float
        measure of thermocline slope. b = 2.5*mu, the coupling coefficient
    R : float
        describes Bjerknes positive feedback process.
    gamma : float
        feedback of the thermocline gradient on the SST.
    en : float
        controls the degree of non-linearity.
    eta1 : float
        Wind stress forcing.
    varyeta1 : bool, optional
        Whether to add wind stress forcing. The default is False.
    varymu : bool, optional
        Whether to vary mu on an annual cycle. The default is False.
    f_ran_forcing : bool, optional
        If varyeta1, whether to include random wind stress forcing.
        The default is True.
    f_ann_forcing : bool, optional
        If varyeta1, whether to include annual wind stress forcing.
        The default is True.
    varyeta2 : float, optional
        Magnitude of random heating. The default is 0.
    W_list : Array, optional
        List of random values between -1 and 1, of length at least 
        endtime/tau_corr. Needed if varyeta1 = True.
        
    Returns
    -------
    T_new : float
        T at the next time step.
    h_new : float
        h at the next time step.

    '''
    if varyeta1:
        #selecting Ws to use in calculations of eta
        W1 = W_list[int(np.floor(time/tau_cor))]
        W12 = W_list[int(np.floor((time+dt/2)/tau_cor))]
        W3 = W_list[int(np.floor((time+dt)/tau_cor))]
        eta1 = calc_eta1(time, dt, W1, f_ran_forcing = f_ran_forcing, f_ann_forcing = f_ann_forcing)
        eta1_12 = calc_eta1(time+dt/2, dt, W12, f_ran_forcing = f_ran_forcing, f_ann_forcing = f_ann_forcing)
        eta1_3 = calc_eta1(time+dt, dt, W3, f_ran_forcing = f_ran_forcing, f_ann_forcing = f_ann_forcing)
        
    else:
        #constant eta1 = 0
        eta1_12 = eta1
        eta1_3 = eta1
    
    if varymu:
        #calculating values of b and R for varying mu
        b = calc_b(mu_0, time, dt)
        b12 = calc_b(mu_0, time+dt/2, dt)
        b3 = calc_b(mu_0, time+dt, dt)
        R = gamma*b-c
        R12 = gamma*b12-c
        R3 = gamma*b3-c
    else:
        #using constant values for b and R
        b12, b3= b, b
        R12, R3 = R, R
    
    #calculating k1, k2, k3, k4
    k1 = f_TE(h, T, R, gamma, en, b, eta1, varyeta2=varyeta2)
    l1 = f_hw(h, T, r, alpha, b, eta1)
    T1, h1 = T+k1*dt/2, h+l1*dt/2
    
    k2 = f_TE(h1, T1, R12, gamma, en, b12, eta1_12, varyeta2=varyeta2)
    l2 = f_hw(h1, T1, r, alpha, b12, eta1_12)
    T2, h2 = T+k2*dt/2, h+l2*dt/2
        
    k3 = f_TE(h2, T2, R12, gamma, en, b12, eta1_12, varyeta2=varyeta2)
    l3 = f_hw(h2, T2, r, alpha, b12, eta1_12)
    T3, h3 = T+k3*dt, h+l3*dt #not divided by 2 here
    
    k4 = f_TE(h3, T3, R3, gamma, en, b3, eta1_3, varyeta2=varyeta2)
    l4 = f_hw(h3, T3, r, alpha, b3, eta1_3)
    #calculating next time step T and h values
    T_new = T + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    h_new = h + dt/6*(l1 + 2*l2 + 2*l3 + l4)
    return T_new, h_new    


def calc_eta1(time, dt, W, f_ran_forcing = True, f_ann_forcing = True):
    '''
    Calculating wind stress forcing at time t

    Parameters
    ----------
    time : float
        nondimensionalised time.
    dt : float
        nondimensionalised time step.
    W : float between -1 and 1
        Random number between -1 and 1.
    f_ran_forcing : Bool, optional
        Whether to include random component of wind stress forcing.
        The default is True.
    f_ann_forcing : Bool, optional
        Whether to include annual component of wind stress forcing.
        The default is True.

    Returns
    -------
    eta1 : float
        winnd stress forcing at time t.

    '''
    eta1 = f_ann_forcing*f_ann*np.cos(2*np.pi*time/tau) + \
        f_ran_forcing*f_ran*W*tau_cor/dt
    return eta1

def calc_b(mu_0, time, dt):
    '''
    calculating b, measure of thermocline slope, at time t. b is at its 
    maximum in may and its minimum in november.

    Parameters
    ----------
    mu_0 : float
    time : float
        nondimensionalised
    dt : float
        nondimensionalised time step.

    Returns
    -------
    b : float
        measure of thermocline slope at time t.

    '''
    mu = mu_0*(1 + mu_ann*np.cos(2*np.pi*time/tau - 5/6*np.pi))
    b = b0*mu
    return b

    