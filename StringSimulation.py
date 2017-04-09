from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.animation as animation
import time
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from multiprocessing import Queue, Process
import multiprocessing

def worker(indices,positions,velocities, positions2,vel,out_q,m,k,same):
    #print "working"
    imax = 0
    jmax = 0
    emax = -10**10
    n = len(positions2)
    for i in indices:
        #print "working row ", i 
        selfpos = positions[i]
        selfvel = velocities[i]
        for j,pos in enumerate(positions2):
            if  (not same) or (j not in [(i-2) % n,(i-1) % n,i,(i+1) % n,(i+2) % n]):
      #      print selfpos, selfvel,vel[j],pos[j]
                e = m* np.linalg.norm(selfvel - vel[j])**2 - k * np.linalg.norm(selfpos - pos)**2 
                if e > emax:
                    imax,jmax,emax = i,j,e
    #print imax,jmax,emax
    out_q.put((imax,jmax,emax))
#def worker(indices,out_q):
##    print "working"
##    imax = 0
##    jmax = 0
##    emax = -10**10
##    n = len(positions2)
##    for i in indices:
##        print "working row ", i 
##        selfpos = positions[i]
##        selfvel = velocities[i]
##        for j,pos in enumerate(positions2):
##            if  (not same) or (j not in [(i-2) % n,(i-1) % n,i,(i+1) % n,(i+2) % n]):
##      #      print selfpos, selfvel,vel[j],pos[j]
##                e = m* np.linalg.norm(selfvel - vel[j])**2 - k * np.linalg.norm(selfpos - pos)**2 
##                if e > emax:
##                    imax,jmax,emax = i,j,e
##    print imax,jmax,emax
#    out_q.put((1,1,1))
#    return



class Loop:
    """
    A class representing a single closed string subject to a spring-like potential as it evolves in time. Considered to be a section of the bundle
    sigma*(T R^2) for a smooth map sigma:S^1 -> R^3.
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
            self.position = np.zeros(len(self.velocity))
        if self.velocity is None:
            self.velocity = np.zeros(len(self.position))
        self.num_points = len(self.position)
        if self.position.shape != self.velocity.shape:
            raise np.linalg.LinAlgError("mismatched position and velocity input shape")

    def re_init(self): #reset the string to initial configuration
        self.__init__(init_pos = self.initial_position, init_vel = self.initial_velocity)
   
    def rotate(self,li = None,n=1): #rotate the source circles by n elements.
        if li is None:
            li = self.position
        return np.array(np.concatenate([li[n:],li[:n]]))

    def com(self):
        return self.position.mean(axis = 0)

    def normsqr(self,list_of_vectors): #compute the norm square of each vector in a list of list of vectors, and collect them in a single list.
        return np.linalg.norm(list_of_vectors,axis = 1)**2
   
##    def acc_old(self,deriv,secondderiv,static_oscillator_parameter = [0.00,0.03,0.07],drag_coefficient=0): #compute the acceleration of a point in the string subject to self forces, in a fixed exterior potential subect to drag (sea of virtual particle?)
##        free = self.k / self.m * secondderiv
##        #constraint = - np.array(
##        #    map(lambda a,b : map(lambda c,d: c*d   ,a,b),
##        #       np.array(np.array([np.linalg.norm(vel,axis =1)**2 for vel in self.velocity])
##        #                - self.k/self.m*np.array([np.linalg.norm(der,axis =1)**2 for der in deriv])),
##        #        self.position)
##        #)
##        bowl = np.array([[1,0,0]] * len(self.position))*self.h
##        poss = self.position #- [  [[self.h,0,0]] * len(loop) for loop in self.position]
##        return  free - 0*self.velocity#+ [pos.dot([[static_oscillator_parameter[0],0,0],[0,static_oscillator_parameter[1],0],[0,0,static_oscillator_parameter[2]]]) for pos in poss] - drag_coefficient*self.velocity #+ 0*constraint #
    
    def acc(self,positions = None,static_oscillator_parameter = [0.00,0.03,0.07],drag_coefficient=0): #compute the acceleration of a point in the string subject to self forces, in a fixed exterior potential subect to drag (sea of virtual particle?) 
        if positions is None:
            positions = self.position
        free = self.k / self.m *( - 2 * self.position + self.rotate(li = positions, n=1) + self.rotate(li = positions, n=-1))
        return  free - drag_coefficient*self.velocity

    def symplectic_integrate(self,initial_position,initial_velocity,acceleration): # a symplectic integration step
        new_velocity = initial_velocity + self.h * acceleration
        new_position = initial_position + self.h * new_velocity
        return new_position,new_velocity

    def energy(self): #the kinetic, self-potential and fixed external potential of the string. Conserved in physics
        return self.kinetic_energy() + self.potential_energy()

    def kinetic_energy(self): #the kinetic, self-potential and fixed external potential of the string. Conserved in physics
        vs = self.normsqr(self.velocity)
        return 0.5 * self.m *vs.sum()

    def potential_energy(self): #the kinetic, self-potential and fixed external potential of the string. Conserved in physics
        lengths= self.normsqr(self.position - self.rotate(n=1))
        return  0.5*self.k *lengths.sum()

      
    def increment(self,drag): #A time step using symplectic Heun's method
        accE = self.acc(drag_coefficient=drag)
        positionEuler, velocityEuler = self.symplectic_integrate(self.position,self.velocity,accE) #Initial Euler step
        accH = self.acc(positions = positionEuler,drag_coefficient=0.1)
        positionHeun, velocityHeun = self.symplectic_integrate(self.position,self.velocity,0.5 *(accE + accH)) #Heun's step
#        err = min(max(self.compute_state_magnitude(positionHeun - positionEuler,velocityEuler-velocityHeun),0.001),0.1) #difference between Heun and Euler integrations estimate the local truncation error to O(t^3)
#        h_old = self.h
        self.position,self.velocity = positionHeun,velocityHeun

    def slicetest(self,string2):
        rowmax,colmax,energymax = 0,0,-10**10
        for i, row in enumerate(self.position):
            for j,col in enumerate(string2.position):
                e = self.m* np.linalg.norm(self.velocity[i] - string2.velocity[j])**2 - self.k * np.linalg.norm(row - col)**2
                if e > energymax:
                    rowmax,colmax,energymax = i,j,e 
        test = (np.random.rand() < 2 / np.pi * np.arctan(energymax/100))
        if string2 == self:
            rowmax,colmax = sorted([rowmax,colmax])
        return test,rowmax,colmax


    def multislicetest(self,string2):
        jobs = []
        q = multiprocessing.Queue()
        posses = self.position
        velles = self.velocity
        posses2 = string2.position
        velles2 = string2.velocity
        n = len(self.position)
        list_of_groups_of_indices = [range(0,len(self.position))]
        for indices in list_of_groups_of_indices:
            p = Process(target=worker,args=(indices,posses, velles,posses2, velles2, q,self.m,self.k,self==string2,))
            jobs.append(p)
            p.start()
        tab = []

        for j in jobs:
            j.join()
            
        for i in range(len(jobs)):
            tab.append(q.get())

        rowmax,colmax,energymax = max(tab,key = lambda a: a[2])
        test = (np.random.rand() < 2 / np.pi * np.arctan(energymax/10))
        if string2 == self:
            rowmax,colmax = sorted([rowmax,colmax])
        return test,rowmax,colmax
        


class StringSystem:
    def __init__(self,strings,h=0.01):
        self.strings = strings
        self.intialstrings = strings
        self.h = h
        for string in self.strings:
            string.h = h #spring constant per unit angle of the string
        self.init_energy = self.energy()

    def re_init(self): #reset the string to initial configuration
        self.__init__(initialsstrings)

    def remove(self,string):
        self.strings.remove(string)
   
    def energy(self): #the kinetic, self-potential and fixed external potential of the string. Conserved in physics
        energy = 0
        for string in self.strings:
            energy += string.energy()
        return energy
   
    def com(self):
        return np.array([string.com() for string in self.strings]).mean(axis=0)

    def increment(self): #A time step using symplectic Heun's method
        for string in self.strings:
            string.increment(-(self.init_energy-self.energy())/self.init_energy)
        if True:
            self.interact()
        print (self.init_energy - self.energy())/self.init_energy
        #print len(self.strings)

    def interact(self,interaction_distance = 0.0,interaction_velocity = 0.5): ##a pair of loops interact with likelihood proportional to one over the sqaure of the number of loops
        #e = self.energy()
        for index1,string1 in enumerate(self.strings):
            for string2 in self.strings[index1:]:
                test,m,n = string1.multislicetest(string2)
                #print string1.slicetest(string2) == string1.multislicetest(string2)
                dec_m = (m - 1) % string1.num_points
                dec_n = (n - 1) % string2.num_points
                #print dec_m,m, len(string1.position),dec_n,n, len(string2.position)
                x1,x2 = string1.position[dec_m],string1.position[m]
                x3,x4 = string2.position[dec_n],string2.position[n]
                v1,v2 = string1.velocity[dec_m],string1.velocity[m]
                v3,v4 = string2.velocity[dec_n],string2.velocity[n]
                PE0 = string1.k/2 * (np.linalg.norm(x1-x2)**2+np.linalg.norm(x3-x4)**2)   
                PE1 = string1.k/2 * (np.linalg.norm(x1-x4)**2+np.linalg.norm(x2-x3)**2)
                KE0 = string1.m/2 * (np.linalg.norm(v1)**2+np.linalg.norm(v2)**2+np.linalg.norm(v3)**2+np.linalg.norm(v4)**2)
                if test and (KE0 + PE0)-PE1 > 0:
                    string1.velocity[m] *= np.sqrt(1-(PE1-PE0)/KE0)
                    string1.velocity[dec_m] *= np.sqrt(1-(PE1-PE0)/KE0)
                    string2.velocity[n] *= np.sqrt(1-(PE1-PE0)/KE0)
                    string2.velocity[dec_n] *= np.sqrt(1-(PE1-PE0)/KE0)
                    if string1 == string2:
                        pos1 = np.concatenate([string1.position[n:],string1.position[:m]])
                        pos2 = string1.position[m:n]
                        vel1 = np.concatenate([string1.velocity[n:],string1.velocity[:m]])
                        vel2 = string1.velocity[m:n]
                        new_string1 =  Loop(init_pos = pos1 ,init_vel = vel1, h = self.h,k=string1.k)
                        new_string2 =  Loop(init_pos =pos2,init_vel = vel2,h = self.h,k=string1.k)
                        self.strings.append(new_string1)
                        self.strings.append(new_string2)
                        self.remove(string1) 
                    else:
                        #print "Strings Merging!"
                        self.remove(string1)
                        self.remove(string2)
                        pos = np.concatenate([string1.position[:m],string2.position[n:],string2.position[:n],string1.position[m:]])
                        vel = np.concatenate([string1.velocity[:m],string2.velocity[n:],string2.velocity[:n],string1.velocity[m:]])
                        new_string = Loop(init_pos = pos, init_vel = vel, h = self.h)
                        self.strings.append(new_string)
                    #print e - self.energy()
                    return
                    
    def run_sim(self):
        while True:
            self.increment()

    
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
            x,y = (x).transpose()
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
            data = np.concatenate([string.position for string in self.strings]).transpose()
            for i in range(1,20):
                self.f[20-i] = self.f[19-i]
            self.f[0] = ax.plot(data[0], data[1], data[2], '.',color=(0.5+0.5*np.sin(2 * np.pi * num/100),0.5+0.5*np.sin(2 * np.pi * num/100 + 2*np.pi/3), 0.5+0.5*np.sin(2 * np.pi * num/100 - 2*np.pi/3)) , lw=2)[0]
            if self.f[19]:
                (self.f[19]).remove()
            xx,yy,zz = self.com()
            ax.set_xlim3d([-2.0+xx,2.0 + xx])
            ax.set_ylim3d([-2.0+yy, 2.0+yy])
            ax.set_zlim3d([-2.0+zz, 2.0+zz])
            return self.f
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        data = np.concatenate([string.position for string in self.strings]).transpose()
        
        # NOTE: Can't pass empty arrays into 3d version of plot()
        self.f = [ax.plot(data[0], data[1], data[2], 'go', lw=2)[0]] + [None for i in range(49)]

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
        Writer = animation.writers['imagemagick_file']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        line_ani.save('lines.mp4', writer=writer)


###
#        Examples
###
def main():
    components = 3
    points = 200.

    possss = (
        5*np.array([np.array(
        [np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
         np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
         np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components)]).transpose()])
        +5*np.array([np.array(
        [np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
         np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
         np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components)]).transpose()])
        )[0]

    velsss = (np.array(np.array([np.array(
        [np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
         np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
         np.cos(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components)]).transpose()]))+
        np.array(np.array([np.array(
        [np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components ),
         np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components),
         np.sin(np.array([np.arange(components)]).transpose().dot(np.array([2*np.pi*np.arange(0,1,1/points)]))).transpose().dot(np.random.rand(components)/components-.5/components)]).transpose()]))
        )[0]

    sting = Loop(init_pos = possss, init_vel = velsss,h = 0.005,k=3)
    global System
    System= StringSystem([sting])
    print [len(string.position) for string in System.strings]
    #System.run_sim()
    System.ThreeDplotter()

if __name__ == "__main__":
    main()
