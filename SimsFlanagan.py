import numpy as np
import pykep as pk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import AstroLibraries.AstroLib_Basic as AL_BF # Imported in AstroLib_2BP
from AstroLibraries import AstroLib_2BP as AL_2BP
from AstroLibraries import AstroLib_Ephem as AL_Eph
from AstroLibraries import AstroLib_Plots as AL_Plot

import LoadConfigFiles as CONFIG
CONF_C = CONFIG.Fitness_config()
CONF = CONF_C.Fit_config

class SimsFlanagan:
    def __init__(self, *args, **kwargs):

        #Constants used
        Cts = AL_BF.ConstantsBook() #Constants for planets

        # Define bodies involved
        self.Spacecraft = AL_2BP.Spacecraft( )
        self.earthephem = pk.planet.jpl_lp('earth')
        self.marsephem = pk.planet.jpl_lp('mars')
        self.jupiterephem = pk.planet.jpl_lp('jupiter')
        self.venusephem = pk.planet.jpl_lp('venus')

        self.sun = AL_2BP.Body('Sun', 'yellow', mu = Cts.mu_S_m)
        self.earth = AL_2BP.Body('Earth', 'blue', mu = Cts.mu_E_m)
        self.mars = AL_2BP.Body('Mars', 'red', mu = Cts.mu_M_m)
        self.jupiter = AL_2BP.Body('Jupiter', 'green', mu = Cts.mu_J_m)
        self.venus = AL_2BP.Body('Venus', 'brown', mu = Cts.mu_V_m)

        departure = kwargs.get('departure', "Earth")
        arrival = kwargs.get('arrival', "Mars")
       
        if departure == 'Earth':
            self.departure = self.earth
            self.departureephem = self.earthephem
        elif departure == 'Mars':
            self.departure = self.mars
            self.departureephem = self.marsephem
        
        if arrival == 'Mars':
            self.arrival = self.mars
            self.arrivalephem = self.marsephem
        elif arrival == 'Jupiter':
            self.arrival = self.jupiter
            self.arrivalephem = self.jupiterephem
        elif arrival == 'Venus':
            self.arrival = self.venus
            self.arrivalephem = self.venusephem
        elif arrival == 'Earth':
            self.arrival = self.earth
            self.arrivalephem = self.earthephem

        # Settings
        self.color = ['blue','green','black','red','yellow','orange']    

        # Choose odd number to compare easily in middle point
        self.Nimp = kwargs.get('Nimp', 11) 

    def adaptDecisionVector(self, DecV, optMode = True):
        """ 
        adaptDecisionVector: modify decision vector to input in the problem
        """
        self.DecV = DecV
        v0 = np.array(DecV[0:3]) # vector, [magnitude, angle, angle]
        vf = np.array(DecV[3:6]) # vector, [magnitude, angle, angle]
        self.t0, self.t_t = DecV[6:8]

        # Delta V
        if optMode == True:
            DeltaV_list = np.array(DecV[8:]).reshape(-1,3) # make a Nimp x 3 matrix
        else:
            DeltaV_list = DecV[8:][0]

        ########################################################################
        # INITIAL CALCULATION OF VARIABLES
        ########################################################################
        # Modify from magnitude angle angle to cartesian
        self.DeltaV_list = np.zeros(np.shape(DeltaV_list))
        DeltaV_sum = np.zeros(len(self.DeltaV_list)) # Magnitude of the impulses
        for i in range(len(self.DeltaV_list)):
            self.DeltaV_list[i, :] = AL_BF.convert3dvector(DeltaV_list[i,:], "polar")
            DeltaV_sum[i] = np.linalg.norm( self.DeltaV_list[i] )

        # Write velocity as x,y,z vector
        v0_cart = AL_BF.convert3dvector(v0, "polar")
        vf_cart = AL_BF.convert3dvector(vf, "polar")


        # Sum of all the Delta V = DeltaV max * sum Delta V_list
        # Assumption: the DeltaV_max is calculated as if mass is constant and 
        # equal to the dry mass as it is the largest contribution. This means 
        # that the DelaV_max is actually smaller than it will be obtained in this
        # problem
        # = Thrust for segment
        self.DeltaV_max = self.Spacecraft.T / self.Spacecraft.m_dry * \
            self.t_t / (self.Nimp + 1) 

        # Total DeltaV 
        DeltaV_total = sum(DeltaV_sum) * self.DeltaV_max

        #Calculate total mass of fuel for the given impulses
        self.m0 = \
            self.Spacecraft.MassChangeInverse(self.Spacecraft.m_dry, DeltaV_total)
        self.m_fuel = self.m0 - self.Spacecraft.m_dry

        # Times and ephemeris
        # t_0 = AL_Eph.DateConv(self.date0,'calendar') #To JD
        self.t_1 = AL_Eph.DateConv(self.t0 + AL_BF.sec2days(self.t_t), 'JD_0' )

        self.r_p0, self.v_p0 = self.departureephem.eph(self.t0)
        self.r_p1, self.v_p1 = self.arrivalephem.eph(self.t_1.JD_0)
        
        # Change from relative to heliocentric velocity
        self.v0 = v0_cart + self.v_p0 
        self.vf = vf_cart + self.v_p1 

        # Create state vector for initial and final point
        self.SV_0 = np.append(self.r_p0, self.v0)
        self.SV_f = np.append(self.r_p1, self.vf) # - to propagate backwards
        self.SV_f_corrected = np.append(self.r_p1, -self.vf) # - to propagate backwards


    def objFunction(self, Error, m_fuel = False, typeError ='vec'):
        
        if typeError == 'vec':
            fc1 = np.linalg.norm(Error[0:3] / AL_BF.AU) # Normalize with AU
            fc2 = np.linalg.norm(Error[3:] / AL_BF.AU * AL_BF.year2sec(1))
            fc1_SI =  np.linalg.norm(Error[0:3])
            fc2_SI = np.linalg.norm(Error[3:])
        else:
            fc1 = Error[0]/ AL_BF.AU
            fc2 = Error[1]/AL_BF.AU*AL_BF.year2sec(1)
            fc1_SI = fc1
            fc2_SI = fc2

        self.Epnorm_norm = fc1
        self.Evnorm_norm = fc2
        self.Epnorm = fc1_SI
        self.Evnorm = fc2_SI

        factor_pos = CONF['FEASIB']['factor_pos']
        factor_vel = CONF['FEASIB']['factor_vel']
        factor_mass = CONF['FEASIB']['factor_mass']


        # if CONF['FEASIB']['log'] == 1:
        #     fc1 = np.log10(fc1)
        #     fc2 = np.log10(fc2)
        #     factor_pos = np.log10(factor_pos)
        #     factor_vel = np.log10(factor_vel)

            

        # print('------------------------')
        # print(m_fuel, Error)
        # print(fc1, fc2)
        if type(m_fuel) == bool:
            value = fc1 * factor_pos + \
                    fc2 * factor_vel
        else:
            fc0 = m_fuel / self.Spacecraft.m_dry

            # else: 
            res = [fc0, fc1, fc2]
            res2 = [fc0, fc1_SI, fc2_SI]
                # print(res)
            
            # print('###################')
            # print(fc1,factor_pos*fc1)
            # print(fc2, factor_vel*fc2)
            # print(fc0, factor_mass*fc0)

            # print(res)
            value = res[0]* factor_mass +\
                    res[1]* factor_pos + \
                    res[2]* factor_vel  # Has to be in AU for the coeff to work in balance
        
        return value


    def calculateFitness(self, DecV, optMode = True, thrust = 'free', 
        printValue = False, plot = False, plot3D = False, 
        massInFunct = True):
        """
        calculateFitness: obtain the value of the fitness function
        INPUTS:
            DecV_I: decision vector of the inner loop
                t_t: transfer time for the leg. In seconds
                DeltaV_list: list of size = Number of impulses 
                            with the 3D impulse between 0 and 1 
                            (will be multiplied by the max DeltaV)
            thrust = free: direction determined by two angles in heliocentric frame.
                    tangential: thrust applied in the direction of motion
            massInFunct: if true, objective function with mass. Otherwise just errors
        """
        
        ########################################################################
        # DecV
        ########################################################################
        self.adaptDecisionVector(DecV, optMode=optMode)

        ########################################################################
        # Propagation
        ########################################################################

        # Sims-Flanagan
        # if plot == True or plot3D == True:
        #     saveState = True
        # else:
        #     saveState = False

        saveState = True

        SV_list_forw = self.__SimsFlanagan(self.SV_0, saveState=saveState, thrust = thrust)
        SV_list_back = self.__SimsFlanagan(self.SV_f_corrected, backwards = True,\
             saveState=saveState, thrust = thrust)

        # convert back propagation so that the signs of velocity match
        SV_list_back_corrected = np.copy(SV_list_back)
        SV_list_back_corrected[:,3:] *= -1 # change sign of velocity

        ########################################################################
        # Compare State in middle point
        ########################################################################
        # print("Error middle point", SV_list_back[-1, :],SV_list_forw[-1, :])
        self.Error = SV_list_back_corrected[-1, :] - SV_list_forw[-1, :]

        if massInFunct == True:
            self.f = self.objFunction(self.Error, m_fuel = self.m_fuel)
        else:
            self.f = self.objFunction(self.Error)

        if printValue == True:
            print("Value: ", self.f)

        if plot == True:
            self.plot2D(SV_list_forw, SV_list_back, [self.sun, self.departure, self.arrival])
        if plot3D == True:
            self.plot3D(SV_list_forw, SV_list_back, [self.sun, self.departure, self.arrival])

        return self.f # *1000 so that in 
                                            

    def calculateMass(self, DecV, optMode = True):
        self.adaptDecisionVector(DecV, optMode=optMode)

        return self.m_fuel


    def __SimsFlanagan(self, SV_i, thrust = 'free',  backwards = False, 
    saveState = False):
        """
        __SimsFlanagan:
            No impulse is not applied in the state 0 or final.
            Only in intermediate steps
        """
        t_i = self.t_t / (self.Nimp + 1) # Divide time in 6 segments

        if saveState == True: # save each state 
            SV_list = np.zeros(((self.Nimp + 1)//2 + 1, 6))
            SV_list[0, :] = SV_i

        # Propagate only until middle point to save computation time
        # ! Problems if Nimp is even
        for imp in range((self.Nimp + 1)//2): 
            # Create trajectory 
            trajectory = AL_2BP.BodyOrbit(SV_i, 'Cartesian', self.sun)
            
            # Propagate the given time 
            # Coordinates after propagation
            SV_i = trajectory.Propagation(t_i, 'Cartesian') 

            # Add impulse if not in last point
            if imp != (self.Nimp + 1)//2:
                if thrust == 'free':
                    if backwards == True:
                        SV_i[3:] -= self.DeltaV_list[imp, :] * self.DeltaV_max # Reduce the impulses
                    else:
                        SV_i[3:] += self.DeltaV_list[imp, :] * self.DeltaV_max
                elif thrust == 'tangential':
                    # If magnitude is backwards:
                    sign = np.sign( np.dot(SV_i[3:], self.DeltaV_list[imp, :]) )
                    if backwards == True:
                        SV_i[3:] -= SV_i[3:]/np.linalg.norm(SV_i[3:])* sign* np.linalg.norm(self.DeltaV_list[imp, :]) * self.DeltaV_max # Reduce the impulses
                    else:
                        SV_i[3:] += SV_i[3:]/np.linalg.norm(SV_i[3:])* sign* np.linalg.norm(self.DeltaV_list[imp, :]) * self.DeltaV_max

            else: # To compare, only add impulse on the one forward to compare delta v
                 if backwards != True:
                        SV_i[3:] += self.DeltaV_list[imp, :] * self.DeltaV_max # Reduce the impulses

            if saveState == True:
                SV_list[imp + 1, :] = SV_i

        if saveState == True:
            return SV_list # All states
        else:
            return SV_i # Last state
    
    def printResult(self):
        print("Mass of fuel", self.m_fuel, "Error", self.Error)

    def DecV2inputV(self, typeinputs, newDecV = 0):

        if type(newDecV) != int:
            self.adaptDecisionVector(newDecV)

        inputs = np.zeros(8)
        inputs[0] = self.t_t
        inputs[1] = self.m0

        if typeinputs == "deltakeplerian":

            # Elements of the spacecraft
            earth_elem = AL_2BP.BodyOrbit(self.SV_0, "Cartesian", self.sun)
            elem_0 = earth_elem.KeplerElem
            mars_elem = AL_2BP.BodyOrbit(self.SV_f, "Cartesian", self.sun)
            elem_f = mars_elem.KeplerElem

            K_0 = np.array(elem_0)
            K_f = np.array(elem_f)

            # Mean anomaly to true anomaly
            K_0[-1] = AL_2BP.Kepler(K_0[-1], K_0[1], 'Mean')[0]
            K_f[-1] = AL_2BP.Kepler(K_f[-1], K_f[1], 'Mean')[0]

            inputs[2:] = K_f - K_0
            inputs[2] = abs(inputs[2]) # absolute value
            inputs[3] = abs(inputs[3]) # absolute value
            inputs[4] = np.cos(inputs[4])# cosine

            labels = [r'$t_t$', r'$m_0$', r'$\vert \Delta a \vert$', r'$\vert \Delta e \vert$',
                     r'$cos( \Delta i) $', r'$ \Delta \Omega$', r'$ \Delta \omega$',
                     r'$ \Delta \theta$']

        elif typeinputs == "deltakeplerian_planet":
            t_1 = self.t0 + AL_BF.sec2days(self.t_t)

            # Elements of the planets
            elem_0 = self.departureephem.osculating_elements(pk.epoch(self.t0, 'mjd2000') )
            elem_f = self.arrivalephem.osculating_elements(pk.epoch(t_1, 'mjd2000'))

            K_0 = np.array(elem_0)
            K_f = np.array(elem_f)

            # Mean anomaly to true anomaly
            K_0[-1] = AL_2BP.Kepler(K_0[-1], K_0[1], 'Mean')[0]
            K_f[-1] = AL_2BP.Kepler(K_f[-1], K_f[1], 'Mean')[0]

            inputs[2:] = K_f - K_0
            inputs[2] = abs(inputs[2]) # absolute value
            inputs[3] = abs(inputs[3]) # absolute value
            inputs[4] = np.cos(inputs[4])# cosine

            labels = [r'$t_t$', r'$m_0$', r'$\vert \Delta a \vert$', r'$\vert \Delta e \vert$',
                     r'$cos( \Delta i) $', r'$ \Delta \Omega$', r'$ \Delta \omega$',
                     r'$ \Delta \theta$']
        
        elif typeinputs == "cartesian":
            delta_r = np.array(self.SV_f[0:3]) - np.array(self.SV_0[0:3])
            delta_v = np.array(self.SV_f[3:]) - np.array(self.SV_0[3:])  
            inputs[2:5] = np.abs(delta_r)
            inputs[5:] = np.abs(delta_v)

            labels = [r'$t_t$', r'$m_0$', r'$\vert \Delta x \vert$', r'$\vert \Delta y \vert$',
                     r'$\vert \Delta z \vert$',r'$\vert \Delta v_x \vert$', r'$\vert \Delta v_y \vert$',
                     r'$\vert \Delta v_z \vert$']

        return inputs, labels
    
    def studyFeasibility(self):
        
        if np.linalg.norm(self.Error[0:3]) <= CONF['FEASIB']['feas_ep'] and \
            np.linalg.norm(self.Error[3:]) <= CONF['FEASIB']['feas_ev']: 
            feasible = 1
        else:
            feasible = 0
        return feasible

    def savetoFile(self, typeinputs, feasibilityFileName, inputs = False):
        """
        savetoFile: save input parameters for neural network and the fitness and
        feasibility

        Append to file

            max m0 should be added
        INPUTS: 
            typeinputs: "deltakeplerian" or "cartesian"
                        save in different columns. 
                deltakeplerian:
                1: label: 0 unfeasible, 1 feasible
                2: t_t: transfer time in s
                3: m0: initial mass of the spacecraft
                4: difference in semi-major axis of the origin and goal
                5: difference in eccentricity of the origin and goal
                6: cosine of the difference in inclination
                7: difference in RAANs
                8: difference in omega
                9: difference in true anomaly
                cartesian:
                1: label: 0 unfeasible, 1 feasible
                2: t_t: transfer time in s
                3: m0: initial mass of the spacecraft
                4: delta x
                5: delta y
                6: delta z
                7: delta vx including the velocity of the spacecraft              
                8: delta vy                
                9: delta vz 
        """
        # Inputs 
        if type(inputs) == bool:
            inputs, labels = self.DecV2inputV(typeinputs)
        else:
            inputs, labels = self.DecV2inputV(typeinputs, newDecV = inputs)
        
        # Feasibility
        feasible = self.studyFeasibility()
        feasible = np.append(feasible, self.m_fuel)

        # Write to file
        vectorFeasibility = np.append(feasible, self.Error)
        vectorFeasibility = np.append(vectorFeasibility, inputs)
        vectorFeasibility = np.append(vectorFeasibility, self.DecV)
        
        with open(feasibilityFileName, "a") as myfile:
            for v, value in enumerate(vectorFeasibility):
                if v != len(vectorFeasibility)-1:
                    myfile.write(str(value) +" ")
                else:
                    myfile.write(str(value))
            myfile.write("\n")
        myfile.close()

    def plot3D(self, SV_f, SV_b, bodies, *args, **kwargs):

        """

        """
        # Create more points for display
        points = 10
        state_f = np.zeros(( (points+1)*(self.Nimp +1)//2 +1, 6 ))
        state_b = np.zeros(( (points+1)*(self.Nimp +1)//2 +1, 6 ))
        t_i = self.t_t / (self.Nimp +1) / (points+1)

        # Change velocity backwards to propagate backwards
        for j in range((self.Nimp+1)//2):
            state_f[(points+1)*j,:] = SV_f[j,:]
            state_b[(points+1)*j,:] = SV_b[j,:]
            for i in range(points):
                trajectory = AL_2BP.BodyOrbit(state_f[(points+1)*j + i], 'Cartesian', self.sun)
                state_f[(points+1)*j + i+1,:] = trajectory.Propagation(t_i, 'Cartesian')

                trajectory_b = AL_2BP.BodyOrbit(state_b[(points+1)*j + i,:], 'Cartesian', self.sun)
                state_b[(points+1)*j + i+1,:] = trajectory_b.Propagation(t_i, 'Cartesian') 

                # trajectory_b = AL_2BP.BodyOrbit(SV_b[j,:], 'Cartesian', self.sun)
                # state_b[(points+1)*j + i+1,:] = trajectory_b.Propagation(t_i*(i+1), 'Cartesian') 

        state_f[-1,:] = SV_f[-1,:]
        state_b[-1,:] = SV_b[-1,:]

        # Plot
        fig = plt.figure()
        ax = Axes3D(fig)
        # ax.view_init(azim=0, elev=10)
        # plot planets
        ax.scatter(0, 0, 0,  color = bodies[0].color, marker = 'o', s = 180, alpha = 0.5)
        ax.scatter(SV_f[0,0], SV_f[0,1], SV_f[0,2], c = bodies[1].color, marker = 'o', s = 150, alpha = 0.5)
        ax.scatter(SV_b[0,0], SV_b[0,1], SV_b[0,2], c = bodies[2].color, marker = 'o', s = 150, alpha = 0.5)

        # xy_annot = ()
        ax.text2D(0.01, 0.90,
                'Error in position: %0.2E m\nError in velocity: %0.2E m/s\nMass of fuel: %i kg'%(self.Epnorm, self.Evnorm, self.m_fuel),
                transform =ax.transAxes)
        # plt.annotate(, xy_annot
        plt.legend([bodies[0].name, bodies[1].name, bodies[2].name])
        # plt.legend(['Error in position: %0.2E m'%self.Epnorm_norm, 'Error in velocity: %0.2E m/s'%self.Evnorm_norm] )

        # plot points for Sims-Flanagan
        x_f = SV_f[:,0]
        y_f = SV_f[:,1]
        z_f = SV_f[:,2]

        x_b = SV_b[:,0]
        y_b = SV_b[:,1]
        z_b = SV_b[:,2]

        ax.scatter(x_f, y_f, z_f, '^-', c = bodies[1].color)
        ax.scatter(x_b, y_b, z_b, '^-', c = bodies[2].color)

        # ax.scatter(x_f[-1],y_f[-1],z_f[-1], '^',c = bodies[1].color)
        # ax.scatter(x_b[-1],y_b[-1],z_b[-1], '^',c = bodies[2].color)

        # ax.plot(state_f[:,0],state_f[:,1], state_f[:,2], 'x-',color = bodies[1].color)
        # ax.plot(state_b[:,0],state_b[:,1], state_b[:,2], 'x-',color = bodies[2].color)
        ax.plot(state_f[:,0],state_f[:,1], state_f[:,2], color = bodies[1].color)
        ax.plot(state_b[:,0],state_b[:,1], state_b[:,2], color = bodies[2].color)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')

        # Plot settings
        AL_Plot.set_axes_equal(ax)

        dpi = kwargs.get('dpi', 200) 
        layoutSave = kwargs.get('layout', 'tight')
        plt.savefig('OptSol/resultopt3D.png', dpi = dpi, bbox_inches = layoutSave)

        plt.show()

    def plot2D(self, SV_f, SV_b, bodies, *args, **kwargs):
        # fig = plt.figure()
        fig, ax = plt.subplots()
        
        # plot planets
        ax.scatter(0, 0,  color = bodies[0].color, marker = 'o', s = 180, alpha = 0.5, label = bodies[0].name)
        ax.scatter(SV_f[0,0], SV_f[0,1], c = bodies[1].color, marker = 'o', s = 150, alpha = 0.5, label = bodies[1].name)
        ax.scatter(SV_b[0,0], SV_b[0,1], c = bodies[2].color, marker = 'o', s = 150, alpha = 0.5, label = bodies[2].name)

        x_f = SV_f[:,0]
        y_f = SV_f[:,1]
        z_f = SV_f[:,2]

        x_b = SV_b[:,0]
        y_b = SV_b[:,1]
        z_b = SV_b[:,2]

        dv1 = self.DeltaV_list[0:len(x_f), :]*1e11
        dv2 = self.DeltaV_list[1:len(x_b), :]*1e11

        for i in range(len(dv1)):
            plt.arrow(x_f[i], y_f[i], dv1[i, 0], dv1[i, 1], length_includes_head=True,
          head_width=5e9, head_length=5e9, fc='k', ec='k')
        for i in range(len(dv2)):
            plt.arrow(x_b[i], y_b[i], dv2[i, 0], dv2[i, 1], length_includes_head=True,
          head_width=5e9, head_length=5e9, fc='k', ec='k')

        ax.plot(x_f, y_f, '^-', c = bodies[1].color)
        ax.plot(x_b, y_b, '^-', c = bodies[2].color)

        ax.scatter(x_f[-1], y_f[-1], marker ='^', c = bodies[1].color)
        ax.scatter(x_b[-1], y_b[-1], marker ='^', c = bodies[2].color)

        plt.axis('equal')
        plt.grid(alpha = 0.5)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
        # AL_Plot.set_axes_equal(ax)

        dpi = kwargs.get('dpi', 200) 
        layoutSave = kwargs.get('layout', 'tight')
        plt.savefig('OptSol/resultopt.png', dpi = dpi, bbox_inches = layoutSave)

        plt.show()

    def plot_tvsT(self):

        t = np.linspace(self.t0, self.t0 +AL_BF.sec2days(self.t_t), num = self.Nimp+2)
        deltaV_i = [np.linalg.norm(self.DeltaV_list[i,:]) *self.DeltaV_max for i in range(len(self.DeltaV_list[:,0])) ]
        deltaV = np.zeros(self.Nimp +2)
        deltaV[1:-1] = deltaV_i 

        fig, ax = plt.subplots()
        plt.plot(t, deltaV, 'o-', color = 'k')

        plt.title("Epoch vs Delta V")
        plt.ylabel("Delta V (m/s)")
        plt.xlabel("JD0 (days)")
        plt.grid(alpha = 0.5)
        # AL_Plot.set_axes_equal(ax)

        dpi = 200
        layoutSave = 'tight'
        plt.savefig('OptSol/tvsT.png', dpi = dpi, bbox_inches = layoutSave)

        plt.show()