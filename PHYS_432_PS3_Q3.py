# -*- coding: utf-8 -*-
"""
@Author: Samy Boutros

@Collaborator: Guilherme H. Caumo

02/28/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg  # Import scipy.linalg for solving tridiagonal systems

n = 1000  # Number of grid points
h = 1  # Lava Layer height (m)
delta_y = h/n  # Spatial step size
t_f = 5  # Final time
steps = 1000  # Number of time steps
delta_t = t_f/steps  #Time step size

# Gravity for inclined slope
g = 9.81  # m/s^2
theta = 5  # Inclination in degrees
rad = theta*np.pi/180 # Inclination in radians
g_slope = g*np.sin(rad)  # Component of gravity along the slope

v = 1  # Normalize viscosity 
diff_coeff = v * delta_t/delta_y**2  # Diffusion coefficient

y = np.linspace(0, h, n)
u_0 = np.zeros(n)  # Initial speed of lava = 0

# Steady-state solution
u_steady = -(g/v)*np.sin(rad)*(y**2/2 - h*y)

plt.ion()
fig, ax = plt.subplots(1,1)

# Setting variables to be updated
pl, = ax.plot(y, u_0, color="red", label = "Varying Flow Velocity")  

# Steady-state solution
ax.plot(y, u_steady, "k", label = "Steady-state solution")

ax.set_title("Lava Flow down $5\degree$ Inclination")
ax.set_ylabel("Flow Speed $u$ (m/s)")
ax.set_xlabel("Lava Height (m)")

plt.legend()

fig.canvas.draw()

#(Used PHYS512 Diffusion notes)

# Define the coefficients for the tridiagonal matrix representing the discretized system
a = -diff_coeff * np.ones(n)  # Lower diagonal
b = (1+2*diff_coeff)*np.ones(n)  # Main diagonal
c = - diff_coeff * np.ones(n)  # Upper diagonal

# Applying BCs to matrix coeffs
a[0] = 0  # No-slip boundary condition at the bottom
b[-1] = 1 + diff_coeff  # Stress-free boundary condition at the top
c[-1] = 0

# Creating matrix
A = np.row_stack((a, b, c))

# Time evolution
for i in range(1, steps):
    # Solving for next time step
    u_0 = scipy.linalg.solve_banded((1, 1), A, u_0 + delta_t*g_slope)
    
    # Update speed
    pl.set_ydata(u_0)
    fig.canvas.draw()
    plt.pause(0.001)  
    
# Keep the plot open after the simulation is done
plt.show()