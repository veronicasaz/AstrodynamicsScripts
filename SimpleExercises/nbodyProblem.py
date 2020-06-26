import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spy
import matplotlib.animation as animation

G = 6.674e-11 #N*m^2/kg^2

class Body:
    def __init__(self, name, mass, radius,r0,v0, color):
        self.name = name
        self.m = mass
        self.R = radius
        self.r0 = r0
        self.v0 = v0
        self.color = color


# #### Bodies
Earth = Body('Earth', 5.9724e24, 6371.0e3,np.array([0,152.10e9,0]),np.array([0,29.78e3,0]),'blue') #m, m/s
Sun = Body('Sun', 1.989e30, 695508e3, np.array([1e9,-1e9,0]),np.array([1e4,0,0]),'orange') #m, m/s
Mars = Body('Mars', 0.64171e24, 3389.5e3, np.array([0,-228e9,0]), np.array([24.07e3,0,0]),'red') #m, m/s
Mercury = Body('Mercury', 3.3011e23, 3389.5e3, np.array([0,57909e6,0]), np.array([47e3,0,0]), 'grey') #m, m/s


Bodies = [Sun, Earth, Mars, Mercury]


X0 = np.zeros(2*3*len(Bodies))
Y = np.zeros(2*3*len(Bodies))

ii = 0
while ii < 3*(len(Bodies)):

    if ii == (3*(len(Bodies))-3):
        X0[ii:ii+3] = Bodies[ii//3].r0
        X0[ii+len(X0)//2:] = Bodies[ii//3].v0

    else:
        X0[ii:ii+3] = Bodies[ii//3].r0
        X0[ii+len(X0)//2:ii+len(X0)//2+3] = Bodies[ii//3].v0
    
    ii += 3



def acceleration(Bodies,X0,r):
    a = np.zeros(3)
    ii = 0
    while ii < 3*(len(Bodies)):

        r2 = X0[ii:ii+3]
        if np.linalg.norm(r2-r) == 0:
             a += np.array([0,0,0])
        else:
            a += G*(Bodies[ii//3].m*(r2-r)) / np.linalg.norm(r2-r)**3
            
#         print(G*(Bodies[ii//3].m*(r2-r)))
#         print('a',a)
        ii += 3
    return a



def solver(X0,t):
    Y[0:len(Y)//2] = X0[len(X0)//2:] # Velocities match
    
    ii = 0
    while ii < 3*(len(Bodies)):
        if ii == (3*(len(Bodies))-3):
            Y[ii+len(Y)//2:] = acceleration(Bodies,X0,X0[ii:ii+3])

        else:
            Y[ii+len(Y)//2:ii+len(Y)//2+3] = acceleration(Bodies,X0,X0[ii:ii+3])
#             print('Y',Y)
        ii += 3
        
    return Y



# # Invariable plane of Laplace
# H = np.zeros(len(Bodies))
# ii = 0
#     while ii < (len(Bodies)):
#         if ii == (3*(len(Bodies))-3):
#             H = 

#         ii += 3


t = np.linspace(0, 600*24*3600, num=50) # Seconds
X = spy.odeint(solver,X0,t)


# plt.plot(x,y,c="red",label="Elliptical orbit")#Ellipse

color = ["orange",'blue','red','grey']

plt.scatter(0,0,c="black",marker="+",label='CM')
plt.grid(True)
plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
    
plt.axis('equal')

plt.xlim([-1e12,1e12])
plt.ylim([-1e12,1e12])
plt.title("Motion of bodies (n-body problem)")

fig = plt.figure()

for i in range(len(t)):
    plt.scatter(0,0,c="black",marker="+",label='CM')
    plt.grid(True)
    plt.grid(color = '0.5', linestyle = '--', linewidth = 1)

    plt.axis('equal')

    plt.xlim([-1e12,1e12])
    plt.ylim([-1e12,1e12])
    plt.title("Motion of bodies (n-body problem)")
    
    ii = 0
    while ii < 3*(len(Bodies)):
        if t[i] == 0:
            plt.scatter(X[i,ii], X[i,ii+1], c=color[ii//3],marker="o",label=Bodies[ii//3].name)
            
        else:
            plt.scatter(X[i,ii], X[i,ii+1], c=color[ii//3],marker="o")
            
        ii += 3
    
    plt.legend(loc = 'upper left') #Position of legend
    # plt.savefig('Name'+str(i)+".png")
    plt.show(block = False)
    plt.close()


# class AnimatedPlot(object):
#     def __init__(self, bodies, X):

#         self.bodies = bodies
#         self.X = X 

#         self.stream = self.data_stream() 

#         self.fig, self.ax = plt.subplots()

#         # Then setup FuncAnimation.
#         self.ani = animation.FuncAnimation(self.fig, self.update, frames = 30, interval=100, 
#                                           init_func=self.setup_plot, blit=True)

#     def setup_plot(self):
#         """Initial drawing of the scatter plot."""

#         x, y, c = next(self.stream).T

#         print(x, y, c)
#         self.scat = self.ax.scatter(x, y, c=c)

#         return self.scat,

#     def data_stream(self):
#         """Generate data"""
#         iterat = 0
#         while True:
#             x = []
#             y = []
#             print(iterat)
#             for i in range(len(self.bodies)):
#                 x.append(X[iterat, 3*i])
#                 y.append(X[iterat, 3*i+1])

#             c = color
#             iterat += 1 
#             yield np.c_[x, y, c]

#     def update(self, i):
#         """Update the scatter plot."""
#         data = next(self.stream)

#         # # Set x and y data...
#         # self.scat.set_offsets(data[:, :2])
#         # # Set sizes...
#         # self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
#         # # Set colors..
#         # self.scat.set_array(data[:, 3])
#         return self.scat,


# anim = AnimatedPlot(Bodies, X)
# plt.show()




