from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.animation as animation
import time
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3



class QString:
    """
    A class representing a number of strings subject to a spring-like potential as it evolves in time. Strings interact, merging and splitting stochastically where they approach one another. At each time step, the string may be considered to be a section of the bundle
    sigma*(T R^2) for a smooth map sigma:(S^1)^n -> R^2, where n is the number of strings at that time.
    """
    def __init__(self,
                 h=0.01,    # the integration time step
                 init_pos = None, # a list of lists giving the location of strings sampled at regular intervals around the source S^1
                 init_vel = None, # a list of lists giving the local velocity of strings sampled at regular intervals around the source S^1
                 m = 1., #mass per unit angle of the string.
                 k = 1.): #spring constant per unit angle of the string
        self.m = m
        self.k = k
        self.time_elapsed = 0.
        self.position = init_pos
        self.velocity = init_vel
        self.initial_position = init_pos
        self.initial_velocity = init_vel
        self.h = h 
        #Next, some sanity checks on input.
        if self.position is None:
            self.position = np.array([np.zeros(pos.shape)*2.*np.pi / len(pos) for pos in self.velocity])
        if self.velocity is None:
            self.velocity = np.array([np.zeros(pos.shape)*2.*np.pi / len(pos) for pos in self.position])
        self.num_points = len(np.concatenate(self.position))
        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])
        if self.position.shape != self.velocity.shape:
            raise np.linalg.LinAlgError("mismatched position and velocity input shape")

    def re_init(self): #reset the string to initial configuration
        self.__init__(init_pos = self.initial_position, init_vel = self.initial_velocity)
   
    def rotate(self,lists = None,n=1): #rotate the source circles by n elements.
        if lists is None:
            lists = self.position
        return np.array([np.concatenate((pos[n:],pos[:n])) for pos in lists])

    def normsqr(self,list_of_list_of_vectors): #compute the norm square of each vector in a list of list of vectors, and collect them in a single list.
        return np.concatenate(np.array([np.linalg.norm(list_of_vectors,axis = 1)**2 for list_of_vectors in list_of_list_of_vectors]))
                
    def compute_deriv(self,positions=None,epsilons=None): #compute the first and second derivatives of the position of a 
        if positions is None: #hack so default values can be attributes of self
            positions = self.position
        if epsilons is None:
            epsilons = self.eps
        yminus=self.rotate(n=-1, lists = positions)  
        y, eps = positions, epsilons
        yplus =self.rotate(n=1,  lists = positions)
        return ( 1/(2 * eps) * ((yplus - y) + (y - yminus)), #first derivative
                1/(eps**2) * ( (yplus - y) - (y - yminus)) ) #second
   
    def acc(self,deriv,secondderiv,static_oscillator_parameter = [0.00,0.03,0.07],drag_coefficient=0): #compute the acceleration of a point in the string subject to self forces, in a fixed exterior potential subect to drag (sea of virtual particle?)
        free = self.k / self.m * secondderiv
        #constraint = - np.array(
        #    map(lambda a,b : map(lambda c,d: c*d   ,a,b),
        #        np.array(np.array([np.linalg.norm(vel,axis =1)**2 for vel in self.velocity])
        #                - self.k/self.m*np.array([np.linalg.norm(der,axis =1)**2 for der in deriv])),
        #        self.position)
        #)
        #bowl = np.array([np.array([[1,0,0]] * len(loop)) for loop in self.position])*10*self.h
        poss = self.position #- [  [[self.h,0,0]] * len(loop) for loop in self.position]
        return  free  - [pos.dot([[static_oscillator_parameter[0],0,0],[0,static_oscillator_parameter[1],0],[0,0,static_oscillator_parameter[2]]]) for pos in poss] - drag_coefficient*self.velocity #+ 0*constraint #
    
    def integrate(self,initial_position,initial_velocity,velocity,acceleration): #an integration step
        new_velocity = initial_velocity + self.h * acceleration
        new_position = initial_position + self.h * velocity
        return new_position,new_velocity

    def symplectic_integrate(self,initial_position,initial_velocity,acceleration): # a symplectic integration step
        new_velocity = initial_velocity + self.h * acceleration
        new_position = initial_position + self.h * new_velocity
        return new_position,new_velocity

    def energy(self): #the kinetic, self-potential and fixed external potential of the string. Conserved in physics
        vs,lengths,poss = self.normsqr(self.velocity),self.normsqr(self.compute_deriv()[0]),self.normsqr(self.position)
        eps = np.concatenate(self.eps)
        return 2 * np.pi *( (0.5 * self.m *eps.transpose().dot(vs)).sum() + 0.5*(self.k *eps.transpose().dot(lengths)).sum()+ 0.5*(0.1 *eps.transpose().dot(poss)).sum())

    def com(self): #the centre of mass of the system. Not conserved unless the exterior potential is zero
        com = (np.concatenate(self.position).sum(axis = 0)/self.num_points)
        return com

    def vocom(self): #the velocity of the  centre of mass of the system. Not conserved unless the exterior potential is zero
        vocom = (np.concatenate(self.velocity).sum(axis = 0)/self.num_points)
        return vocom


    def compute_state_magnitude(self,pos,vel): #A measure of the size of a state, in length units. The intention is to use this for adaptive step sizes
        return (max(self.m /self.k * self.normsqr(vel) + self.normsqr(pos)))**(0.5)
        
    def increment(self): #A time step using symplectic Heun's method
        deriv,secondderiv = self.compute_deriv(positions = self.position)
        accE = self.acc(deriv,secondderiv)
        positionEuler, velocityEuler = self.symplectic_integrate(self.position,self.velocity,accE) #Initial Euler step
        deriv,secondderiv = self.compute_deriv(positions=positionEuler) 
        accH = self.acc(deriv,secondderiv)
        positionHeun, velocityHeun = self.symplectic_integrate(self.position,self.velocity,0.5 *(accE + accH)) #Heun's step
#        err = min(max(self.compute_state_magnitude(positionHeun - positionEuler,velocityEuler-velocityHeun),0.001),0.1) #difference between Heun and Euler integrations estimate the local truncation error to O(t^3)
#        h_old = self.h
        self.position,self.velocity = positionHeun,velocityHeun
        if np.random.rand()/self.num_points**2 < .0005: #allow self interactions randomly, at most one per time step.
            self.interaction(interaction_distance = 0.1,interaction_velocity=0.3)
#        return self.position , self.velocity


    def interaction(self,interaction_distance = 0.1,interaction_velocity = 0.5): ##a pair of loops interact with likelihood proportional to one over the sqaure of the number of loops
        out = []
        outvel = []        
        loop_index_1,loop_index_2 =0,np.random.randint((len(self.position)*(len(self.position)+1))/2)
        while loop_index_2 > loop_index_1: #select a pair of loops, (1,1) as likely as (1,2)=(2,1)
            loop_index_1 += 1
            loop_index_2 -= loop_index_1
        for loop_index,loops in enumerate(self.position):
            loopvels = self.velocity[loop_index]
            if loop_index != loop_index_1 and loop_index != loop_index_2: #throw all the other loops into the output untouched
                out.append(loops)  
                outvel.append(loopvels)
        if loop_index_1 == loop_index_2: #loop self-interaction
            loop = self.position[loop_index_1]
            loopvel = self.velocity[loop_index_1]
            loop_processed = False
            if len(loop) > 6: #short loops don't self-interact
                a = np.random.randint(len(loop)) 
                b = np.random.randint(len(loop))
                if a < b:
                    element_index,other_element_index = a,b #pick two points on the loop to potentially interact
                else:
                    element_index,other_element_index = b,a
                if True:#min(abs(other_element_index-element_index),abs(other_element_index-len(loop)-element_index))>max(5,len(loop)/3):#something just to make the string kinda long and to prevent too-easy self-interaction:
                    if (min(abs(other_element_index-element_index),abs(other_element_index-len(loop)-element_index))>6
                        and np.linalg.norm(loopvel[element_index] - loopvel[other_element_index])*np.random.rand() > interaction_velocity
                        and np.linalg.norm(loop[element_index] - loop[other_element_index]) < interaction_distance):
                        #points must be nearby and moving fast relative to each other in order to interact
                        print "INTERACTION!"
                        print "now " + str(len(self.position) + 1) + " loops"
                        #cut the loop into two, same with the velocity tensor for the loop
                        loop1,loopvel1 = (np.concatenate((   loop[other_element_index:],   loop[:element_index])),
                                          np.concatenate((loopvel[other_element_index:],loopvel[:element_index])))
                        loop2,loopvel2 = (   loop[element_index:other_element_index],
                                          loopvel[element_index:other_element_index])
                        out.append(loop1)
                        out.append(loop2)
                        outvel.append(loopvel1)
                        outvel.append(loopvel2)
                        loop_processed = True
            if not loop_processed: #if we didn't cut the loop, add it to the output.
                out.append(loop)
                outvel.append(loopvel)
        else: #loops different
            loop_1,loopvel_1 = self.position[loop_index_1],self.velocity[loop_index_1]
            loop_2,loopvel_2 = self.position[loop_index_2],self.velocity[loop_index_2]
            element_index,other_element_index = np.random.randint(len(loop_1)),np.random.randint(len(loop_2))#pick points on each interacting loop
            if (np.linalg.norm(loopvel_1[element_index] - loopvel_2[other_element_index])*np.random.rand() > interaction_velocity
                and np.linalg.norm(loop_1[element_index] - loop_2[other_element_index]) < interaction_distance):
                #points must be nearby and moving fast relative to each other in order to interact
                print "INTERACTION!"
                print "now " + str(len(self.position) - 1) + " loops"
                #join these loops
                loop,loopvel = (
                    np.concatenate((   loop_1[:element_index],   loop_2[other_element_index:],   loop_2[:other_element_index],   loop_1[element_index:])),
                    np.concatenate((loopvel_1[:element_index],loopvel_2[other_element_index:],loopvel_2[:other_element_index],loopvel_1[element_index:])))
                out.append(loop)
                outvel.append(loopvel)  
            else:   #if the loops didn't merge after all.
                out.append(loop_1)
                outvel.append(loopvel_1) 
                out.append(loop_2)
                outvel.append(loopvel_2)
        self.position = np.array(out) #update data
        self.velocity = np.array(outvel)    
        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])  #update the source circle angle spacing data
                    

    def plotter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-3, 3), ylim=(-3,3))
        ax.grid()
        line, = ax.plot([], [], 'bo', lw=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        other_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        com_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)        

        def init():
            """initialize animation"""
            line.set_data([], [])
            time_text.set_text('')
            energy_text.set_text('')
            com_text.set_text('') 
            return line, time_text, energy_text, com_text

        def animate(i):
            """perform animation step"""
            self.increment()
            #for j,line in enumerate(lines):
            x,y = (np.concatenate(self.position)).transpose()
            line.set_data(x,y)
            #time_text.set_text('curve = ' + str(string.curvature))
            time_text.set_text('time = ' + str(self.time_elapsed))
            energy_text.set_text('energy = %.3f J' % self.energy())
            com_text.set_text('COM = ' + str(self.com()))
            other_text.set_text('Loop Count = ' + str(self.position.shape[0]))

            return line,  time_text, energy_text,other_text,com_text

        from time import time
        t0 = time()
        animate(0)
        t1 = time()
        interval =(100 * self.h - (t1 - t0))
        print interval

        ani = animation.FuncAnimation(fig, animate, frames=300,
                                      interval=interval, blit=True, init_func=init)

        plt.show()

    def ThreeDplotter(self):

        def update_lines(num):
            for i in range(10):
                self.increment()
            data = np.concatenate(self.position).transpose()
            for i in range(1,50):
                self.f[50-i] = self.f[49-i]
            self.f[0] = ax.plot(data[0], data[1], data[2], 'o',color=(1, 0.75+ 0.2*np.sin(2 * np.pi * num/100 + 2*np.pi/3),0.5 + 0.5* np.sin(2 * np.pi * num/100)) , lw=2)[0]
            if self.f[49]:
                (self.f[49]).remove()
            xx,yy,zz = self.com()
            ax.set_xlim3d([-2.0+xx,2.0 + xx])
            ax.set_ylim3d([-2.0+yy, 2.0+yy])
            ax.set_zlim3d([-2.0+zz, 2.0+zz])
            ax.set_xlim3d([-2.0+ 10*num*self.h,2.0 + 10*num*self.h])
            ax.set_ylim3d([-2.0, 2.0])
            ax.set_zlim3d([-2.0, 2.0])

                        #    ax.collections.remove(self.frame)
            #self.frame = ax.plot(data[0], data[1], data[2], '.',color=(1, 0.5+ 0.5*np.sin(2 * np.pi * num/100 + 2*np.pi/3),0* np.sin(2 * np.pi * num/100)**2) , lw=2)
            return self.f
        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = p3.Axes3D(fig)


        # Fifty lines of random 3-D lines
        data = np.concatenate(self.position).transpose()
        # NOTE: Can't pass empty arrays into 3d version of plot()
        self.f = [ax.plot(data[0], data[1], data[2], 'go', lw=2)[0]] + [None for i in range(49)]
##        self.ff= None
##        self.fff= None
##        self.ffff= None
##        self.fffff= None
##        self.ffffff= None
##        self.fffffff= None
##        self.ffffffff= None
##        self.fffffffff= None
##        self.ffffffffff= None
        # Setting the axes properties
        ax.set_xlim3d([-2.0,2.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-2.0, 2.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-2.0, 2.0])
        ax.set_zlabel('Z')

        ax.set_title('3D Test')

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, update_lines, range(10000),
                                           interval=1, blit=False)

        plt.show()



###
#        Examples
###
poss,vels = (
    np.array([np.array([
        cos(np.arange(0,1,0.0025) * 2 * np.pi),
        sin(np.arange(0,1,0.0025) * 2 * np.pi)]).transpose()]),
    np.array([0.5*np.pi* np.array([1*sin(np.arange(0,1,0.0025) * 3* 2 * np.pi),
                cos(np.arange(0,1,0.0025)*2 * 2 * np.pi)]).transpose()]))
vels = vels - vels.sum(0)/len(vels)
string = QString(init_pos = poss)
    
ttring = QString(init_pos = poss)
  
pos = 5*np.array([np.array(
    [np.cos(
        np.array([np.arange(10)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))
        ).transpose().dot(np.random.rand(10)/10-.5/10),
     np.cos(np.array([np.arange(10)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(10)/10-.5/10)]).transpose()])+poss
vel = np.array(np.
               array([np.array(
    [np.cos(np.array([np.arange(10)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(10)/10-.5/10),
     np.sin(np.array([np.arange(10)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(10)/10-.5/10)]).transpose()]))

sstring = QString(init_pos = pos,init_vel = vel)

pos = (
    5*np.array([np.array(
    [np.cos(np.array([np.arange(4)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(4)/4-.5/3),
     np.cos(np.array([np.arange(4)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(4)/4-.5/3)]).transpose()])
    +5*np.array([np.array(
    [np.sin(np.array([np.arange(4)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(4)/4-.5/3),
     np.sin(np.array([np.arange(4)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(4)/4-.5/3)]).transpose()])
    +2*poss)


vel = (np.array(np.array([np.array(
    [np.cos(np.array([np.arange(4)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(4)/4-.5/3),
     np.cos(np.array([np.arange(4)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(4)/4-.5/3)]).transpose()]))+
    np.array(np.array([np.array(
    [np.sin(np.array([np.arange(4)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(4)/4-.5/3),
     np.sin(np.array([np.arange(4)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(4)/4-.5/3)]).transpose()]))
       +2*vels)

s = QString(init_pos = pos,init_vel = vel)


pos = np.array([
    np.array([cos(np.arange(0,1,0.01) * 2 * np.pi),
              sin(np.arange(0,1,0.01) * 2 * np.pi),
              cos(np.arange(0,1,0.01) * 2 * np.pi)]).transpose(),
    ])

ssttring = QString(init_pos = pos,h=0.01)
points = 12.0
poss,vels = (
    np.array([np.array([
        cos(np.arange(0,1,1/points) * 2 * np.pi),
        sin(np.arange(0,1,1/points) * 2 * np.pi)]).transpose()]),
    2 *np.pi * 0* np.array([np.pi* np.array([-sin(np.arange(0,1,1/points) *  2 * np.pi),
                cos(np.arange(0,1,1/points) * 2 * np.pi)]).transpose()]))

sting = QString(init_pos = poss, init_vel = vels,h = 0.1)

theta = 10*np.cos(np.array([np.arange(4)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,0.0025)]))).transpose().dot(np.random.rand(4)/4-.5/3) +2*np.pi*np.arange(0,1,0.0025)

on_circle = np.array([[ np.cos(theta),np.sin(theta)]]).transpose((0,2,1))
string_on_circle = QString(init_pos = on_circle)

components = 3
points = 400.

pos = (
    5*np.array([np.array(
    [np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
     np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
     np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components)]).transpose()])
    +5*np.array([np.array(
    [np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
     np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
     np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components)]).transpose()])
    )


vel = (np.array(np.array([np.array(
    [np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components)+0.95,
     np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
     np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components)]).transpose()]))+
    np.array(np.array([np.array(
    [np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
     np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
     np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components)]).transpose()]))
    )

sting = QString(init_pos = pos, init_vel = vel,h = 0.01,k=0.03)

