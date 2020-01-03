import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

from AstroLib_Plots import Plot, Subplot

pl = Plot('Test', nrows = 2, mcols = 2)
pl.createPlot()

x = np.arange(0, 100,1)
y = x*2

plS = Subplot('Test0', 'Test0', type_plot = 'scatter')
plS.size(titleSize = 20)

plS2 = Subplot('Test1', 'Test1', type_plot = 'scatter')
plS2.size(titleSize = 5)

pl.addSubplot(plS, x, y, 1, xlabel = 'x label', markers = ['o'], grid = True)
pl.addSubplot(plS2, x, y, 2, xlabel = 'x label', markers = ['x'], grid = True)

pl.show()