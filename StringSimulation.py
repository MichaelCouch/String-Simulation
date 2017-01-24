from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.animation as animation
import time


num_steps  = 1000 # number of time intervals
total_time = 10.
hh = total_time / num_steps
planck_length = 0.01

class QString:

    def __init__(self,
                 h=0.01,
                 init_pos = None,
                 init_vel = None,
                 m = 1.,
                 k = 1.):
        self.m = m
        self.k = k
        self.time_elapsed = 0.
        self.position = init_pos
        self.velocity = init_vel
        self.initial_position = init_pos
        self.initial_velocity = init_vel
        self.h = h
        if self.position is None:
            self.position = np.array([np.zeros(pos.shape)*2.*np.pi / len(pos) for pos in self.velocity])
        if self.velocity is None:
            self.velocity = np.array([np.zeros(pos.shape)*2.*np.pi / len(pos) for pos in self.position])
        self.num_points = len(np.concatenate(self.position))
        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])
        if self.position.shape != self.velocity.shape:
            raise np.linalg.LinAlgError("mismatched position and velocity input shape")

    def re_init(self):
        self.__init__(init_pos = self.initial_position, init_vel = self.initial_velocity)
   
    def rotate(self,lists = None,n=1):
        if lists is None:
            lists = self.position
        return np.array([np.concatenate((pos[n:],pos[:n])) for pos in lists])

    def normsqr(self,list_of_list_of_vectors):
        return np.concatenate(np.array([np.linalg.norm(list_of_vectors,axis = 1)**2 for list_of_vectors in list_of_list_of_vectors]))
                
    def compute_deriv(self,positions=None,epsilons=None):
        if positions is None:
            positions = self.position
        if epsilons is None:
            epsilons = self.eps
        yminus=self.rotate(n=-1,lists = positions)  
        y, eps = positions, epsilons
        yplus =self.rotate(n=1,lists = positions)
        return ( 1/(2 * eps) * ((yplus - y) + (y - yminus)),
                1/(eps**2) * ( (yplus - y) - (y - yminus)) )        

   
    def acc(self,deriv,secondderiv,static_oscillator_parameter = 0.1,drag_coefficient=0):
        return self.k / self.m * secondderiv - static_oscillator_parameter * self.position - drag_coefficient*self.velocity

#        acc = np.array(
#            map(lambda dd,d,de: dd, #dd/de - dd.dot(d) * d / (de**3) ,
#                extext,    ext,   np.linalg.norm(ext,axis=1)))
#        return acc
            


    def integrate(self,initial_position,initial_velocity,velocity,acceleration):
        new_position = initial_position + self.h * velocity
        new_velocity = initial_velocity + self.h * acceleration
        return new_position,new_velocity


    def energy(self):
        vs,lengths,poss = self.normsqr(self.velocity),self.normsqr(self.compute_deriv()[0]),self.normsqr(self.position)
        eps = np.concatenate(self.eps)
        return 2 * np.pi *( (0.5 * self.m *eps.transpose().dot(vs)).sum() + 0.5*(self.k *eps.transpose().dot(lengths)).sum()+ 0.5*(0.1 *eps.transpose().dot(poss)).sum())


    def compute_state_magnitude(self,pos,vel):
        return (max(0.1*self.m /self.k * self.normsqr(vel) + self.normsqr(pos)))**(0.5)
        
    def increment(self):
#        print len(self.position)," loops"
        deriv,secondderiv = self.compute_deriv()
        accE = self.acc(deriv,secondderiv)
        positionEuler, velocityEuler = self.integrate(self.position,self.velocity,self.velocity,accE)
        deriv,secondderiv = self.compute_deriv(positions=positionEuler) 
        accH = self.acc(deriv,secondderiv)
        positionHeun, velocityHeun = self.integrate(self.position,self.velocity,0.5 * (self.velocity + velocityEuler),0.5 *(accE + accH))
        self.position,self.velocity = positionHeun,velocityHeun
        err = self.compute_state_magnitude(positionHeun - positionEuler,velocityEuler-velocityHeun)
#        self.position,self.velocity = positionEuler,velocityEuler
        h_old = self.h
        self.h *= np.sqrt(0.01/err)
        print round(np.log10(self.h))
        if np.random.rand()/self.num_points**2 < .00005:
            self.faster_interaction()
####        self.accel = accH
###     if self.max_move()> .01:
####            self.h = 0.5 * h
####            return self.position , self.velocity
####        if self.max_move()< .003:
####            self.h = 2 * h
        self.time_elapsed += h_old
        #time.sleep(10*self.h)
        if self.time_elapsed > 10:
            self.re_init()
        return self.position , self.velocity


##    def re_sample(self):
##        threshold = .1
##        curvatures = np.linalg.norm(
##                    self.acc,
##                    axis=1)
##        pos = []
##        vel = []
##        eps = []
##        angs = np.cumsum(self.eps)
##        
##        i=0
##        while i < len(curvatures):
##            if curvatures[i]> threshold:
##                j = i
##                while j < len(self.position) and curvatures[j] > threshold:
##                    j += 1
##                    #now j is the first time curvature isnt too large
##                #indexing problem here when we go around the circle
##                if j < len(self.position):
##                    inc_j = (j+1)%len(self.position)
##                    x,y = self.position[i-1:inc_j].transpose()
##                    vx,vy = self.velocity[i-1:inc_j].transpose()
##                    a = angs[i-1:inc_j]
##                    lpos = (scipy.interpolate.lagrange(a,x),scipy.interpolate.lagrange(a,y))
##                    lvel = (scipy.interpolate.lagrange(a,vx),scipy.interpolate.lagrange(a,vy))
##     #          print i,j,range(i-1,j+1),angs[i-1:j+1],self.position[i-1:j+1]
##     #          print np.arange(angs[i-1],angs[j],(angs[j]-angs[i-1])/(2*(j+1-i)))+ (angs[j]-angs[i-1])/(2*(j+1-i))
##                    for ang in np.arange(angs[i-1],angs[j],(angs[j]-angs[i-1])/(2*(j+1-i)))+(angs[j]-angs[i-1])/(2*(j+1-i)):
##     #                  print "angles we interpolate at ",ang, "inserting"
##                        pos.append([lpos[0](ang),lpos[1](ang)])
##     #                  print [lpos[0](ang),lpos[1](ang)]
##                        vel.append([lvel[0](ang),lvel[1](ang)])
##                        eps.append((angs[j]-angs[i-1])*0.5/(j-i+1.))
##                else:
##                    try:
##                        x,y = np.concatenate((self.position[i-1:],self.position[:1])).transpose()
##                        vx,vy = np.concatenate((self.velocity[i-1:],self.velocity[:1])).transpose()
##                        a = np.append(angs[i-1:],angs[0]+2 *np.pi)
##                    except ValueError:
##                        print self.position[i-1:].shape,self.position[:1].shape
##                        raise
##                    lpos = (scipy.interpolate.lagrange(a,x),scipy.interpolate.lagrange(a,y))
##                    lvel = (scipy.interpolate.lagrange(a,vx),scipy.interpolate.lagrange(a,vy))
##     #          print i,j,range(i-1,j+1),angs[i-1:j+1],self.position[i-1:j+1]
##     #          print np.arange(angs[i-1],angs[j],(angs[j]-angs[i-1])/(2*(j+1-i)))+ (angs[j]-angs[i-1])/(2*(j+1-i))
##                    for ang in np.arange(angs[i-1],angs[0]+2*np.pi,(angs[0]+2*np.pi-angs[i-1])/(2*(j+1-i)))+(angs[0]+2*np.pi-angs[i-1])/(2*(j+1-i)):
##     #                  print "angles we interpolate at ",ang, "inserting"
##                        pos.append([lpos[0](ang),lpos[1](ang)])
##     #                  print [lpos[0](ang),lpos[1](ang)]
##                        vel.append([lvel[0](ang),lvel[1](ang)])
##                        try:
##                            eps.append((angs[0]+np.pi*2-angs[i-1])*0.5/(j-i+1.))
##                        except:
##                            print j
##                            raise
##                i = j+2                
##            else:
## #              print angs[i], self.position[i]
##                pos.append(self.position[i])
##                vel.append(self.velocity[i])
##                eps.append(self.eps[i])
##                i += 1
##        self.position = np.array(pos)
##        self.velocity = np.array(vel)
##        self.eps = np.array(eps)


##    def interaction(self):
##        out = []
##        outvel = []
##        num_loops = len(self.position)
##        for loop_index in range(num_loops):
##            loop = self.position[loop_index]
##            loopvel = self.velocity[loop_index]
##            loop_processed = False
##            if len(loop) > 6:
##                for element_index in range(1,len(loop)-1,5):
##                    for other_element_index in range(max(element_index+5,element_index + len(loop)/3,5),min(len(loop)-5+element_index,len(loop))):
##                        if type(self.position[0][0][0]) != np.float64:
##                            print "Oh Noes!",element_index, other_element_index,
##                        try:
##                            if np.linalg.norm(loop[element_index] - loop[other_element_index]) <min(np.linalg.norm(loop[element_index] - loop[element_index+1]),0.1):
##                                if np.random.rand() < .005 and not loop_processed:
##                                    loop1 = np.concatenate((loop[other_element_index:],loop[:element_index]))
##                                    loopvel1 = np.concatenate((loopvel[other_element_index:],loopvel[:element_index]))
##                                    loop2 = loop[element_index:other_element_index]
##                                    loopvel2 = loopvel[element_index:other_element_index]
##                                    out.append(loop1)
##                                    outvel.append(loopvel1) 
##                                    out.append(loop2)
##                                    outvel.append(loopvel2)
##                                    loop_processed = True
##                                    print out, outvel,"great!"
##                        except ValueError:
##                           # print loop_index, element_index, other_element_index#, loop[element_index]
##                            raise
##                        if loop_processed:
##                            break
##                    if loop_processed:
##                        break
##            if not loop_processed:
##                out.append(loop)
##                outvel.append(loopvel)
##        self.position = np.array(out)
##        self.velocity = np.array(outvel)
##        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])
##                            
##    def fast_interaction(self):
##        out = []
##        outvel = []
##        num_loops = len(self.position)
##        for loop_index in range(num_loops):
##            loop = self.position[loop_index]
##            loopvel = self.velocity[loop_index]
##            loop_processed = False
##            if len(loop) > 6:
##                element_index = np.random.randint(0,len(loop))
##                for other_element_index in range(max(element_index+5,element_index + len(loop)/3,5),min(len(loop)-5+element_index,len(loop))):
##                    if type(self.position[0][0][0]) != np.float64:
##                        print "Oh Noes!",element_index, other_element_index,
##                    try:
##                        if np.linalg.norm(loop[element_index] - loop[other_element_index]) <min(np.linalg.norm(loop[element_index] - loop[element_index+1]),0.1):
##                            if np.random.rand() < .0000002*len(loop)**2 and not loop_processed:
##                                loop1 = np.concatenate((loop[other_element_index:],loop[:element_index]))
##                                loopvel1 = np.concatenate((loopvel[other_element_index:],loopvel[:element_index]))
##                                loop2 = loop[element_index:other_element_index]
##                                loopvel2 = loopvel[element_index:other_element_index]
##                                out.append(loop1)
##                                outvel.append(loopvel1) 
##                                out.append(loop2)
##                                outvel.append(loopvel2)
##                                loop_processed = True
##                                print out, outvel,"great!"
##                    except ValueError:
##                       # print loop_index, element_index, other_element_index#, loop[element_index]
##                        raise
##                    if loop_processed:
##                        break
##            if not loop_processed:
##                out.append(loop)
##                outvel.append(loopvel)
##        self.position = np.array(out)
##        self.velocity = np.array(outvel)
##        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])

    def faster_interaction(self):
        out = []
        outvel = []
        loop_index_1,loop_index_2 = np.random.randint(0,len(self.position)),np.random.randint(0,len(self.position))
        for loop_index,loops in enumerate(self.position):
            loopvels = self.velocity[loop_index]
            if loop_index != loop_index_1 and loop_index != loop_index_2:
                out.append(loops)
                outvel.append(loopvels)
                #print "added non-interacting loops"
        if loop_index_1 == loop_index_2:
            loop = self.position[loop_index_1]
            loopvel = self.velocity[loop_index_1]
            loop_processed = False
            if len(loop) > 6:
                a = np.random.randint(len(loop))
                b = np.random.randint(len(loop))
                if a < b:
                    element_index,other_element_index = a,b
                else:
                    element_index,other_element_index = b,a
                if min(abs(other_element_index-element_index),abs(other_element_index-len(loop)-element_index))>max(5,len(loop)/3):#something just to make the string kinda long and to prevent too-easy self-interaction:
#                    print "Interaction possible at loop ",loop_index_1," sites ",element_index,other_element_index 
#                    print np.linalg.norm(loop[element_index] - loop[other_element_index])
                    if np.linalg.norm(loop[element_index] - loop[other_element_index]) < min(10*np.linalg.norm(loop[element_index] - loop[(element_index+2)%len(loop)]),0.1):
                        print "INTERACTION!"
                        loop1 = np.concatenate((loop[other_element_index:],loop[:element_index]))
                        loopvel1 = np.concatenate((loopvel[other_element_index:],loopvel[:element_index]))
                        print "1st loop ok"
                        loop2 = loop[element_index:other_element_index]
                        loopvel2 = loopvel[element_index:other_element_index]
                        print "2ns loop ok"
                        out.append(loop1)
                        outvel.append(loopvel1*0.9) #dampen the energy gain    ) 
                        out.append(loop2)
                        outvel.append(loopvel2*0.9) #dampen the energy gain    )
                        print "loops added"
                        loop_processed = True
                        if len(loopvel1) + len(loopvel2) != len(loopvel):
                            raise ValueError("changed number of points in velocity")
            if not loop_processed:
                out.append(loop)
                outvel.append(loopvel)
        else: #loops different
            loop_1 = self.position[loop_index_1]
            loop_2 = self.position[loop_index_2]
            loopvel_1 = self.velocity[loop_index_1]
            loopvel_2 = self.velocity[loop_index_2]
            element_index_1 = np.random.randint(len(loop_1))
            element_index_2 = np.random.randint(len(loop_2))
            try:
                if np.linalg.norm(loop_1[element_index_1] - loop_2[element_index_2]) < min(10*np.linalg.norm(loop_1[element_index_1] - loop_1[(element_index_1+2)%len(loop_1)]),0.1):
                    print "INTERACTION!"
                    loop = np.concatenate((
                        loop_1[:element_index_1],loop_2[element_index_2:],loop_2[:element_index_2],loop_1[element_index_1:]
                        ))
                    loopvel = np.concatenate((
                        loopvel_1[:element_index_1],loopvel_2[element_index_2:],loopvel_2[:element_index_2],loopvel_1[element_index_1:]
                        ))
                    out.append(loop)
                    outvel.append(loopvel*0.9) #dampen the energy gain  
                else:
                    out.append(loop_1)
                    outvel.append(loopvel_1) 
                    out.append(loop_2)
                    outvel.append(loopvel_2)
            except IndexError:
                print loop_index_1,element_index_1,len(loop_1),loop_index_2,element_index_2,len(loop_2)
                raise
        if len(np.concatenate(np.array(outvel))) != self.num_points:
            raise ValueError("You changed the number of points some points, loop indices" + str(loop_index_1) + str(loop_index_2))
        self.position = np.array(out)
        self.velocity = np.array(outvel)    
        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])            
                    

    def plotter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-3, 3), ylim=(-3,3))
        ax.grid()
        line, = ax.plot([], [], 'bo', lw=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        other_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        

        def init():
            """initialize animation"""
            line.set_data([], [])
            time_text.set_text('')
            energy_text.set_text('') 
            return line, time_text, energy_text
##        def init():    
##            for line in lines:
##                line.set_data([], [])
##            return lines

        def animate(i):
            """perform animation step"""
            self.increment()
            #for j,line in enumerate(lines):
            x,y = (np.concatenate(self.position)).transpose()
            line.set_data(x,y)
            #time_text.set_text('curve = ' + str(string.curvature))
            time_text.set_text('time = ' + str(self.time_elapsed))
            energy_text.set_text('energy = %.3f J' % self.energy())
            other_text.set_text('Loop Count = ' + str(self.position.shape[0]))

            return line,  time_text, energy_text,other_text

        from time import time
        t0 = time()
        animate(0)
        t1 = time()
        interval =(100 * self.h - (t1 - t0))
        print interval

        ani = animation.FuncAnimation(fig, animate, frames=300,
                                      interval=interval, blit=True, init_func=init)

        plt.show()

##
##
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
vel = np.array(np.array([np.array(
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
              sin(np.arange(0,1,0.01) * 2 * np.pi)]).transpose(),
    ])

ssttring = QString(init_pos = pos,h=0.01)

