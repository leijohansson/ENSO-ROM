# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:18:04 2024

@author: Linne
"""

import numpy as np
from fixed_constants import *

def f_hw(h_w, T_E, r, alpha, b, eta1):
    f = -r*h_w - alpha*b*T_E - alpha*eta1
    return f

def f_TE(h_w, T_E, R, gamma, en, b, eta1, varyeta2 = False): 
    eta2 = 0
    if varyeta2:
        eta2 = 0.2*(np.random.random()*2-1)
    f = R*T_E + gamma*h_w - en*(h_w + b*T_E)**3 + gamma*eta1 + eta2
    return f
    
def RK4(h, T, dt, time, mu_0, r, alpha, b, R, gamma, en, eta1, varyeta1 = False, varymu = False, f_ran_forcing=True, varyeta2  = False):
    if varyeta1:
        eta1 = calc_eta1(time, dt, f_ran_forcing = f_ran_forcing)
        eta1_12 = calc_eta1(time+dt/2, dt, f_ran_forcing = f_ran_forcing)
        eta1_3 = calc_eta1(time+dt, dt, f_ran_forcing = f_ran_forcing)
    else:
        eta1_12 = eta1
        eta1_3 = eta1
    
    if varymu:
        b = calc_b(mu_0, time, dt)
        b12 = calc_b(mu_0, time+dt/2, dt)
        b3 = calc_b(mu_0, time+dt, dt)
        R = gamma*b-c
        R12 = gamma*b12-c
        R3 = gamma*b3-c
    else:
        b12, b3= b, b
        R12, R3 = R, R
    
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
    
    T_new = T + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    h_new = h + dt/6*(l1 + 2*l2 + 2*l3 + l4)
    return T_new, h_new    

def calc_eta1(time, dt, f_ran_forcing = True):
    W = np.random.random()*2-1
    eta1 = f_ann*np.cos(2*np.pi*time/tau) + f_ran_forcing*f_ran*W*tau_cor/dt
    return eta1
def calc_b(mu_0, time, dt):
    mu = mu_0*(1 + mu_ann*np.cos(2*np.pi*time/tau - 5/6*np.pi))
    b = b0*mu
    return b

    