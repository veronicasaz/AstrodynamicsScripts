import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

from AstroLib_Plots import Plot, Subplot

# pl = Plot('Test', n=1, m=1)

# plt.save('Test.fig')
# pl.show()

x = np.arange(0,100,1)
y = x*2

pl = Subplot('Test0', 'Test0', 1, type_plot = 'scatter')
pl.size(titleSize = 20)
pl.plot(x,y, xlabel = 'x label', markers = ['o'], grid = True)

pl.show()