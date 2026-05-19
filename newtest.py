#%% Header: importing python libraries

import numpy as np  # important package for scientific computing
from scipy import signal  # signal processing library
import matplotlib.pyplot as plt  # library to plot graphics
import vrft  # vrft package

#%% Declarating input signal

# number of samples
N = 100
t = np.linspace(0, N - 1, N)
t = t.reshape((1, N))
# step signal
# u = np.ones((N, 1))
# u[0] = 0

# using a square wave for both inputs
# defining the period of the square wave
ts = N / 2
fs = 1 / ts
# finally, defining the square wave using the function signal.square()
u = 0.5 - 0.5 * signal.square(2 * np.pi * fs * t).T

#%% Td and 1-Td

# transfer function
Td = signal.TransferFunction([0.1], [1, -0.9], dt=1)
Td = [[Td]]
Atd, Btd, Ctd, Dtd = vrft.mtf2ss(Td)
# simulate output from Td
ytd = vrft.filter(Td, u)
# 1-Td
Td2 = signal.TransferFunction([1, -1], [1, -0.9], dt=1)
Td2 = [[Td2]]
Atd2, Btd2, Ctd2, Dtd2 = vrft.mtf2ss(Td2)
ytd2 = vrft.filter(Td2, u)

print(Atd, Btd, Ctd, Dtd)
print(Atd2, Btd2, Ctd2, Dtd2)

uhat1, _, flag = vrft.stbinv(Atd, Btd, Ctd, Dtd, ytd.T, t)
print(u.T.shape)
print(uhat1.shape)

uhat2, _, flag = vrft.stbinv(Atd2, Btd2, Ctd2, Dtd2, ytd2.T, t)
print(uhat2.shape)

#%%
# plot input estimated signals versus real input signal
lw = 1.5