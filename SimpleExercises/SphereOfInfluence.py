#!/usr/bin/env python
# coding: utf-8

# ## 3. Sphere of influence

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spy

G = 6.674e-11 #N*m^2/kg^2


# In[2]:


class Body:
    def __init__(self, mass, radius,r0,v0):
        self.m = mass
        self.R = radius
        self.r0 = r0
        self.v0 = v0


# In[3]:


Earth = Body(5.972e24, 6371.0e3,np.array([152.10e9,0,0]),np.array([0,29.78e3,0])) #m, m/s
Sun = Body(1.989e30, 695508e3, np.array([0,0,0]),np.array([0,0,0])) #m, m/s
Mars = Body(0.64171e24, 3389.5e3, np.array([0,228.0e9,0]), np.array([24.07e3,0,0])) #m, m/s


# #### Comet 67P/C-G: Churyumov–Gerasimenko

# In[11]:


Rosetta = Body(9.982e12, 0, np.array([0,0,0]), np.array([0,0,0])) #m, m/s
r_a = 850150.0e6 #m
r_p = 185980.0e6 #m


# In[5]:


# At perihelium
rho = r_p #m 

R_i_p = rho*(Rosetta.m/Sun.m)**(2/5)
print('Sphere of influence. R_i (perihelium) =',R_i_p/1e3,'km')


# In[7]:


# At apohelium
rho = r_a #m 
R_i_a = rho*(Rosetta.m/Sun.m)**(2/5)
print('Sphere of influence. R_i (apohelium) =',R_i_a/1e3,'km')

