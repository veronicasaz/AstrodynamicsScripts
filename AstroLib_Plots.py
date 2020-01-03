import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

class Plot:
    def __init__(self, titleFig, *args, **kwargs):
        """          
        titleFig: title of the figure
        
        additional arguments:
        """
       # Number of of the subplots
        self.n = kwargs.get('nrows',1)
        self.m = kwargs.get('mcols',1)

        # Size of fig
        self.h = kwargs.get('h',10)
        self.v = kwargs.get('v',15) 

        self.titleFig = titleFig

        # Markers
        self.colors = ['black','red', 'blue', 'green','orange','yellow','magenta','cyan']
        self.marker = ['x','o','-','v','^','s','h','+']
        self.line = ['-','--','-.',':'] 

    def createPlot(self, *args, **kwargs):
        self.titleSize = kwargs.get('titleSize',15)
        
        self.h_space = kwargs.get('hspace',0.4)
        self.w_space = kwargs.get('wspace',0.25)
    
        self.top_margin = kwargs.get('top',0.92)
        self.bot_margin = kwargs.get('bottom',0.08)
        self.right_margin = kwargs.get('right',0.95)
        self.left_margin = kwargs.get('left',0.1)

        # Create fig and subplots
        self.fig, self.ax = plt.subplots(self.n, self.m, figsize=(self.h, self.v))    
        self.fig.suptitle(self.titleFig, size = self.titleSize )
        self.fig.subplots_adjust(top=self.top_margin, bottom=self.bot_margin, left=self.left_margin, right=self.right_margin, hspace=self.h_space, wspace=self.w_space)

        self.listSubplots = dict()

    def addSubplot(self, subplot, x, y, order, *args, **kwargs):
        subplot.plot(x, y, order, args, **kwargs)
        self.listSubplots[subplot.name] = subplot # Create list with the subplots

    def save(self, name, *args, **kwargs):
        self.dpi = kwargs.get('dpi', 200) 
        self.layoutSave = kwargs.get('layout', 'tight')
        plt.savefig(name, dpi = self.dpi, bbox_inches = self.layoutSave)

    def show(self):
        plt.show()


class Subplot(Plot):
    def __init__(self, name, titleFig, *args, **kwargs):
        Plot.__init__(self, name, titleFig, *args, **kwargs)

        # Type plot
        self.name = name
        self.type_plot = kwargs.get('type_plot','other')

        self.titleFig = titleFig

    def size(self, *args, **kwargs):
        # Size
        self.titleSize = kwargs.get('titleSize',15)
        self.axisSize = kwargs.get('axisSize',15)
        self.legendSize = kwargs.get('legendSize',10)
        self.ticksLabel = kwargs.get('ticksLabel',10)

    def plot(self, x, y, order, *args, **kwargs):
        # Data
        self.x = x
        self.y = y
        self.order = order

        # self.sets_xdata = kwargs.get('setsx',1)
        self.sets_ydata = kwargs.get('setsx',1)

        # Plot labels
        self.xlabel = kwargs.get('xlabel', ' ')
        self.ylabel = kwargs.get('ylabel', ' ') 
        self.labels = kwargs.get('labels', [''])

        self.legend = kwargs.get('legend',False)
        self.grid = kwargs.get('grid', False)
        self.gridAlpha = kwargs.get('gridAlpha', 0.5 )

        self.markersSubpl = kwargs.get('markers', self.marker)
        self.colorsSubpl = kwargs.get('colors', self.colors)
        self.lineSubpl = kwargs.get('line', self.line)

        if self.type_plot == 'scatter':
            plt.subplot(self.n, self.m, self.order)

            if self.sets_ydata == 1:
                self.y = [y]

            for i in range(self.sets_ydata):
                plt.scatter(self.x , self.y[i], label = self.labels[i], marker = self.markersSubpl[i] , color = self.colorsSubpl[i])
                
        else:
                plt.plot(x[i,:] , y[i,:] ,label = self.label[i], linestyle = self.line[i] , color = self.color[i])
        

        plt.title(self.titleFig, size = self.titleSize)
        plt.grid(self.grid, alpha = self.gridAlpha)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        if self.legend == 'True':
            plt.legend()


    # def show(self):
    #     plt.show()

        