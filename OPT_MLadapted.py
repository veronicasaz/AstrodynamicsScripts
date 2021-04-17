#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

import time

import AstroLibraries.AstroLib_Basic as AL_BF 

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spy

import time

import AstroLibraries.AstroLib_Basic as AL_BF 

class MyTakeStep(object):
    def __init__(self, Nimp, bnds):
        self.Nimp = Nimp
        self.bnds = bnds
    def __call__(self, x):
        self.call(x)
        return x

    def call(self, x, magnitude):
        np.random.seed(10)
        x2 = np.zeros(len(x))
        for j in range(len(x)):
            x2[j] = x[j] + magnitude* np.random.uniform(self.bnds[j][0] - x[j], \
                                    self.bnds[j][1] - x[j], 1)[0]

        # x[0] += np.random.normal(-1e3, 1e3, 1)[0] # velocity magnitude
        # x[1:3] += np.random.normal(-0.2, 0.2, 1)[0] # angle
        # x[3] += np.random.normal(-1e3, 1e3, 1)[0] # velocity magnitude
        # x[4:6] += np.random.normal(-0.2, 0.2, 1)[0] # angle
        # x[6] += np.random.normal(-30, 30, 1)[0] # time in days
        # x[7] += np.random.normal(-30, 30, 1)[0] # initial mass
        # x[8] += np.random.normal(- AL_BF.days2sec(30), AL_BF.days2sec(30), 1)[0] # transfer time 
        # for i in range(self.Nimp):
        #     x[9+i*3] += np.random.normal(-0.1, 0.1, 1)[0]
        #     x[9+i*3+1 : 9+i*3+3] += np.random.normal(-0.5, 0.5, 2)
        
        return x2

def print_fun(f, x, accepted):
    """
    print_fun: choose what to print after each iteration
    INPUTS: 
        x: current value of the decision vector
        f: function value
        accepted: iteration improves the function = TRUE. Otherwise FALSE.
    OUTPUTS: none
    """
    print("Change iteration",accepted)
    print("#####################################################################")
    # print('t',time.time() - start_time)


def check_feasibility(x, bnds):
    feasible = True
    for j in range(len(x)): 
        if bnds[j][0] <0:
            tol = 1.05
        else:
            tol = 0.95
        if ( x[j] < tol*bnds[j][0] ) or ( x[j] > 1.05*bnds[j][1] ): # 1.05* to have tolerance
            feasible = False
            print(j, "Within bounds?", "min", bnds[j][0], "value",x[j], "max",bnds[j][1])
    # print(feasible)

    return feasible


def seqEval(f, x):
    """
    Evaluate each row of x with f and return vector with result
    """
    sol = np.zeros(len(x[:,0]))
    for i in range(len(x[:,0])):
        sol[i] = f(x[i,:])
    return sol

def verify_ML_result_fit(f,y,x):
    for i in range(len(y)):
        print("Real vs predicted fit", f(x[i,:]), y[i])

def verify_ML_result(f, x, y, best_y, tolLocal, bnds):
    solLocal = spy.minimize(f, x, method = 'SLSQP', \
                                            tol = tolLocal, bounds = bnds )
    y_real = solLocal.fun
    print("Real vs predicted", y_real, y)
    if y_real < best_y:
        return True, y_real
    else:
        return False, y_real


def MonotonicBasinHopping_ML(f, x, take_step, *args, **kwargs):
    """
    Step for jump is small, as minimum cluster together.
    Jump from current min until n_no improve reaches the lim, then the jump is random again.
    """
    f_batch = kwargs.get('f_batch', None) # Function allows for batch eval
    f_opt = kwargs.get('f_opt', None) # function to optimze locally

    if f_opt == None:
        MLlabel = False
    else:
        MLlabel = True

    nind = kwargs.get('nind', 100)
    niter = kwargs.get('niter', 100)
    niter_success = kwargs.get('niter_success', 50)
    bnds = kwargs.get('bnds', None)
    jumpMagnitude_default =  kwargs.get('jumpMagnitude', 0.1) # Small jumps to search around the minimum
    jumpMagnitude_big = kwargs.get('jumpMagnitude_big', 1)
    tolGlobal = kwargs.get('tolGobal', 1e-5)

    pathSave = './OptSol/'
    # if f_opt == None:
    #     saveLabel = 'False'
    # else:
    #     saveLabel = 'True
    niter_local = kwargs.get('niter_local', 50)
    tolLocal = kwargs.get('tolLocal', 1e2)
    time_iter = np.zeros(niter+1)
    n_itercounter = 0
    n_noimprove = np.zeros((nind))

    # Save information
    Best_x = x # Best x vector for each individual
    jumpMagnitude = np.ones((nind)) * jumpMagnitude_default # vector with jump magn
    accepted = np.zeros((nind)) 
    feasibility = np.zeros((nind))

    # Evaluate
    if f_batch != None:
        bestMin = f_batch(x)
        bestMin = bestMin.flatten()
        verify_ML_result_fit(f, bestMin, x)
    else:
        bestMin =  seqEval(f, x) # Fitness for each value of best x
    previousMin = bestMin.copy() # Previous minimum (to know if there is improvement)

    # Save progression in time
    allMin_iter = np.zeros((niter+1, nind)) # best value accumulated
    currentValue_iter = np.zeros((niter+1, nind)) # current obtained value

    allMin_iter[0,:] = bestMin
    currentValue_iter[0, :] = bestMin

    start_time = time.time()
    time_iter[0] = 0
    while n_itercounter < niter:
        n_itercounter += 1
        
        # Change decision vector to find one within the bounds
        x_test = np.zeros(np.shape(x))
        for ind in range(nind):
            feasible = False
            while feasible == False and bnds != None:
                x_test[ind] = take_step.call(x[ind], jumpMagnitude[ind])
                feasible = check_feasibility(x_test[ind], bnds)

        # Local optimization 
        if f_opt != None:
            currentMin = f_opt(x_test)
            currentMin = currentMin.flatten()
            solutionLocal = x_test
            print("CurrentMin", currentMin)
        else:
            currentMin = np.zeros((nind))
            solutionLocal = np.zeros(np.shape(x))
            for ind in range(nind):
                solLocal = spy.minimize(f, x_test[ind], method = 'SLSQP', \
                                            tol = tolLocal, bounds = bnds, 
                                            options = {'maxiter': niter_local} )
                solutionLocal[ind] = solLocal.x

                feasibility[ind] = check_feasibility(solLocal.x, bnds)
                if feasibility[ind] == False:
                    print("Warning: Point out of limits")

            if f_batch != None:
                currentMin = f_batch(solutionLocal)
                currentMin = currentMin.flatten()
            else:
                currentMin = seqEval(f, solutionLocal)

             
        for ind in range(nind):
            # Check te current point from which to jump: after doing a long jump or
            # when the solution is improved        
            if jumpMagnitude[ind] == jumpMagnitude_big or currentMin[ind] < previousMin[ind]:
                x[ind] = solutionLocal[ind] 
                
            # Check improvement 
            # if currentMin[ind] < bestMin[ind] and feasibility[ind] == True: # Improvement 
            if currentMin[ind] < bestMin[ind]: # Improvement   
                # verify result if ML is present in the optimization
                print("current ind min", currentMin[ind], min(bestMin))
                if f_batch != None and currentMin[ind] <= min(bestMin):
                    verified, = verify_ML_result_fit(f, solutionLocal[ind], currentMin[ind], bestMin[ind])
                else:
                    verified = True

                if f_opt != None and currentMin[ind] <= min(bestMin):
                    verified, bestreal = verify_ML_result(f, solutionLocal[ind], currentMin[ind], bestMin[ind], tolLocal, bnds)
                else:
                    verified = True
                    bestreal = currentMin[ind]

                if verified == True:
                    Best_x[ind] = x[ind]
                    bestMin[ind] = bestreal
                    accepted[ind] = True
                    # If the improvement is not large, assume it is not improvement
                    if (previousMin[ind] - currentMin[ind] ) < tolGlobal: 
                        n_noimprove[ind] += 1
                    else:
                        n_noimprove[ind] = 0
                    jumpMagnitude[ind] = jumpMagnitude_default
                else:
                    accepted[ind] = False
                    jumpMagnitude[ind] = jumpMagnitude_default
                    n_noimprove[ind] += 1 

            elif n_noimprove[ind] == niter_success: # Go to a bigger jump
                accepted[ind] = False
                jumpMagnitude[ind] = jumpMagnitude_big
                n_noimprove[ind] = 0 # Restart count so that it performs jumps around a point
            else:
                accepted[ind] = False
                jumpMagnitude[ind] = jumpMagnitude_default
                n_noimprove[ind] += 1        
            
        previousMin = currentMin.copy()

        # Save current min
        allMin_iter[n_itercounter,:] = bestMin
        currentValue_iter[n_itercounter, :] = previousMin

        time_iter[n_itercounter] = (time.time() - start_time) 

        # Save results every n iter
        if n_itercounter %5 == 0:
            with open(pathSave +'allMBH_batch_'+str(MLlabel)+'.txt', 'wb') as filename:
                for row in np.matrix(Best_x):
                    np.savetxt(filename, row)

        print("iter", n_itercounter)
        print("Current min vs best one", min(currentMin), min(bestMin))
        print_fun(min(currentMin), min(bestMin), np.count_nonzero(accepted == True))

    mat1 = np.matrix(allMin_iter)
    mat2 = np.matrix(currentValue_iter)
    with open(pathSave +'allMin_'+str(MLlabel)+'.txt', 'wb') as filename:
        for row in mat1:
            np.savetxt(filename, row)
    with open(pathSave +'currentValues_'+str(MLlabel)+'.txt', 'wb') as filename:
        for row in mat2:
            np.savetxt(filename, row)

    with open(pathSave +'time_iter_'+str(MLlabel)+'.txt', 'wb') as writefile:
        np.savetxt(writefile, time_iter)

    # Print solution 
    # print_sol(Best_x, bestMin, n_itercounter, niter_success)
    return Best_x, bestMin
        



def plotConvergence(ML=False):
    allmin = np.loadtxt('./OptSol/allMin_'+str(False)+'.txt')
    time_iter = np.loadtxt('./OptSol/time_iter_'+str(False)+'.txt')

    allmin_ML = np.loadtxt('./OptSol/allMin_'+str(True)+'.txt')
    # currentValues = np.loadtxt('./OptSol/currentValues.txt')
    time_iter_ML = np.loadtxt('./OptSol/time_iter_'+str(True)+'.txt')

    allmin_ML_2 = np.loadtxt('./OptSol/allMin_'+str(True)+'_1000.txt')
    time_iter_ML_2 = np.loadtxt('./OptSol/time_iter_'+str(True)+'_1000.txt')

    color = ('black', 'red', 'green', 'blue', 'orange')

    iterations = len(allmin[:,0])
    individuals = len(allmin[0,:])

    # PLOT 1. CONVERGENCE OF BEST MIN
    bestmin = np.zeros(iterations)
    for i in range(iterations):
        bestmin[i] = min(allmin[i,:])
    bestmin_ML = np.zeros(iterations)
    bestmin_ML_2 = np.zeros(iterations)
    for i in range(iterations):
        bestmin_ML[i] = min(allmin_ML[i,:])
        bestmin_ML_2[i] = min(allmin_ML_2[i,:])
 
    # fig, ax = plt.subplots(2, 1)
    fig, ax = plt.subplots(2, 1, figsize = (15,8), gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    print(len(allmin_ML), len(bestmin_ML) )

    ax[0].plot(np.arange(0, iterations, 1), bestmin, color = 'black', marker =  'o', label = 'Without ML surrogate')
    ax[0].plot(np.arange(0, iterations, 1), bestmin_ML, color = 'blue', marker =  'x', label = 'With ML surrogate (20 ind)')
    ax[0].plot(np.arange(0, iterations, 1), bestmin_ML_2, color = 'green', marker =  '+', label = 'With ML surrogate (1000 ind)')

    # Annotate values
    annotate_points = np.arange(0, iterations, iterations//4)
    for i in annotate_points:
        # ax[0].annotate('%.3E'%bestmin[i], (i, bestmin[i]+1e-1), color = 'black',rotation= 45, fontsize = 13)
        # ax[0].annotate('%.3E'%bestmin_ML[i], (i-3, bestmin_ML[i]+1e-1), color = 'blue', rotation= 45, fontsize = 13)
        # ax[0].annotate('%.3E'%bestmin_ML_2[i], (i-5, bestmin_ML_2[i]*0.6), color = 'green', rotation= 45, fontsize = 13)
        avg= min(bestmin[i], bestmin_ML[i], bestmin_ML_2[i])
        ax[0].annotate('%.3E'%bestmin[i], xy = (i, bestmin[i]), xycoords ='data', 
                xytext = (15,avg -bestmin[i] +70), textcoords = 'offset pixels', 
                arrowprops=dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90, rad = 1", color = 'black'),  color = 'black', fontsize = 13)
        ax[0].annotate('%.3E'%bestmin_ML[i], xy = (i, bestmin_ML[i]), xycoords ='data', 
                xytext = (15,avg -bestmin_ML[i] +50), textcoords = 'offset pixels', 
                arrowprops=dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90, rad = 1", color = 'blue'),  color = 'blue', fontsize = 13)
        ax[0].annotate('%.3E'%bestmin_ML_2[i], xy = (i, bestmin_ML_2[i]), xycoords ='data', 
                xytext = (15,avg -bestmin_ML_2[i]  + 40), textcoords = 'offset pixels', 
                arrowprops=dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90, rad = 1", color = 'green'),  color = 'green', fontsize = 13)
        # ax[0].annotate('%.3E'%bestmin_ML_2[i], (i, bestmin_ML_2[i]), arrowprops=dict(arrowstyle="->",
        #             connectionstyle="angle,angleA=0,angleB=-90,rad=1"), color = 'green', fontsize = 13)
    

    # Fix plot
    ax[0].set_yscale('log')
    ax[0].set_ylim(1e-1,10e2)
    ax[0].set_ylabel("Current Minimum ( J )", fontsize = 18)
    ax[0].legend(fontsize = 18)

    ax[1].plot(np.arange(0, iterations, 1), time_iter, color = 'black', marker =  'o', label = 'Without ML surrogate')
    ax[1].plot(np.arange(0, iterations, 1), time_iter_ML, color = 'blue', marker =  'x', label = 'With ML surrogate (20 ind)')
    ax[1].plot(np.arange(0, iterations, 1), time_iter_ML_2, color = 'green', marker =  '+', label = 'With ML surrogate (1000 ind)')
    
    ax[1].set_ylabel("Computation time (s)", fontsize = 18)

    for axi in ax[0], ax[1]:
        axi.set_xlabel('Iteration', fontsize = 18)
        axi.tick_params(axis='both', which='major', labelsize=14)
        axi.tick_params(axis='both', which='minor', labelsize=12)
        axi.grid(alpha = 0.5)

    plt.suptitle("Monotonic Basin Hopping \nConvergence plot", fontsize = 20)
    plt.savefig("./OptSol/OptimizationConvergence_together.png", dpi = 100)
    plt.show()

    # # PLOT 2. CONVERGENCE OF ALL IND
    # fig, (ax) = plt.subplots(1, 1)
    # fig.subplots_adjust(wspace=0.1, hspace=0.05)

    # displayiter = np.arange(0, iterations , iterations //2)  
    # for i, iteration in enumerate(displayiter):
    #     ax.plot(np.arange(0, individuals , 1), currentValues[iteration, :], color = color[i], marker =  'o', label = 'Iteration: '+str(iteration))

    # ax.set_yscale('log')
    # ax.set_xlabel("Individual")
    # ax.set_ylabel("Current Minimum")
    # ax.grid(alpha = 0.5)
    # ax.legend()

    # plt.savefig("./OptSol/OptimizationConvergence_all.png", dpi = 100)
    # plt.show()



