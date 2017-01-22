from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.animation as animation


num_steps  = 1000 # number of time intervals
total_time = 10.
hh = total_time / num_steps
planck_length = 0.01

class QString:

    def __init__(self,
                 hh=0.01,
                 init_pos = None,
                 init_vel = None,
                 m = 1.,
                 k = 1.):
        self.position = init_pos
        self.velocity = init_vel
        self.h = hh
        if self.position == None:
            self.position = np.array([np.zeros(pos.shape)*2.*np.pi / len(pos) for pos in self.velocity])
        if self.velocity == None:
            self.velocity = np.array([np.zeros(pos.shape)*2.*np.pi / len(pos) for pos in self.position])
        self.m = m
        self.k = k
        self.time_elapsed = 0.
        self.num_points = len(np.concatenate(self.position))
        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])
        if self.position.shape != self.velocity.shape:
            raise np.linalg.LinAlgError("mismatched position and velocity input shape")
#        self.ext,self.extext = self.compute_extension()
        
    def rotate(self,n=1):
        return (np.array([np.concatenate((pos[n:],pos[:n])) for pos in self.position]),
                np.array([np.concatenate((vel[n:],vel[:n])) for vel in self.velocity]),
                np.array([np.concatenate((ha[n:],ha[:n])) for ha in self.eps]))
                
    def rotate_Euler(self,n=1):
        return (np.array([np.concatenate((pos[n:],pos[:n])) for pos in self.positionEuler]),
                np.array([np.concatenate((eps[n:],eps[:n])) for eps in self.eps]))

    def map_level(self,f, item, level):
        if level == 0:
            return f(item)
        else:
            return [self.map_level(f, i, level - 1) for i in item]


    def compute_extension(self):
        yminus, vminus, epsminus=self.rotate(-1)  
        y, eps = self.position, self.eps  
        yplus, vplus, epsplu=self.rotate(1)
        epsplus = np.array([epsplu,epsplu]).transpose()   
#        return  ( 1/(eps + epsplus) * ((yplus - y)*eps/(epsplus) + (y - yminus)*epsplus/eps),
#                    2/(eps + epsplus) * ( (yplus - y)/epsplus - (y - yminus)/eps ) )
        try:
            return ( 1/(2 * eps) * ((yplus - y) + (y - yminus)),
                    1/(eps**2) * ( (yplus - y) - (y - yminus)) )
        except ValueError:
            print y.shape, yplus.shape, yminus.shape,eps.shape
            raise
        
    def compute_extension_Euler(self):
##        yminus, epsminus=self.rotate_Euler(-1)  
##        y, eps = self.positionEuler, np.array([self.eps,self.eps]).transpose()  
##        yplus, epsplu=self.rotate_Euler(1)
##        epsplus = np.array([epsplu,epsplu]).transpose()  
##        return  ( 1/(eps + epsplus) * ((yplus - y)*eps/(epsplus) + (y - yminus)*epsplus/eps),
##                    2/(eps + epsplus) * ( (yplus - y)/epsplus - (y - yminus)/eps ) )
        yminus, epsminus=self.rotate_Euler(-1)  
        y, eps = self.positionEuler, self.eps  
        yplus, epsplu=self.rotate_Euler(1)
        epsplus = np.array([epsplu,epsplu]).transpose()   
#        return  ( 1/(eps + epsplus) * ((yplus - y)*eps/(epsplus) + (y - yminus)*epsplus/eps),
#                    2/(eps + epsplus) * ( (yplus - y)/epsplus - (y - yminus)/eps ) )
        try:
            return ( 1/(2 * eps) * ((yplus - y) + (y - yminus)),
                    1/(eps**2) * ( (yplus - y) - (y - yminus)) )
        except ValueError:
            print y.shape, yplus.shape, yminus.shape,eps.shape
            raise
        
    def acc(self,ext,extext):
#        acc = np.array(
#            map(lambda dd,d,de: dd, #dd/de - dd.dot(d) * d / (de**3) ,
#                extext,    ext,   np.linalg.norm(ext,axis=1)))
#        return acc
        return extext - 0.1 * self.position
#    def posd(self):
#        return np.array(
#            map(lambda a,b: a/b,
#                (self.position - self.rotate(1)[0]),
#                self.eps))
#        posd = np.zeros(self.position.shape)
#        for i in len(self.position):
#            inc_i = (i+1)%len(self.position)
#            posd = self.position
            

    def re_sample(self):
        threshold = .1
        curvatures = np.linalg.norm(
                    self.acc,
                    axis=1)
        pos = []
        vel = []
        eps = []
        angs = np.cumsum(self.eps)
        
        i=0
        while i < len(curvatures):
            if curvatures[i]> threshold:
                j = i
                while j < len(self.position) and curvatures[j] > threshold:
                    j += 1
                    #now j is the first time curvature isnt too large
                #indexing problem here when we go around the circle
                if j < len(self.position):
                    inc_j = (j+1)%len(self.position)
                    x,y = self.position[i-1:inc_j].transpose()
                    vx,vy = self.velocity[i-1:inc_j].transpose()
                    a = angs[i-1:inc_j]
                    lpos = (scipy.interpolate.lagrange(a,x),scipy.interpolate.lagrange(a,y))
                    lvel = (scipy.interpolate.lagrange(a,vx),scipy.interpolate.lagrange(a,vy))
     #          print i,j,range(i-1,j+1),angs[i-1:j+1],self.position[i-1:j+1]
     #          print np.arange(angs[i-1],angs[j],(angs[j]-angs[i-1])/(2*(j+1-i)))+ (angs[j]-angs[i-1])/(2*(j+1-i))
                    for ang in np.arange(angs[i-1],angs[j],(angs[j]-angs[i-1])/(2*(j+1-i)))+(angs[j]-angs[i-1])/(2*(j+1-i)):
     #                  print "angles we interpolate at ",ang, "inserting"
                        pos.append([lpos[0](ang),lpos[1](ang)])
     #                  print [lpos[0](ang),lpos[1](ang)]
                        vel.append([lvel[0](ang),lvel[1](ang)])
                        eps.append((angs[j]-angs[i-1])*0.5/(j-i+1.))
                else:
                    try:
                        x,y = np.concatenate((self.position[i-1:],self.position[:1])).transpose()
                        vx,vy = np.concatenate((self.velocity[i-1:],self.velocity[:1])).transpose()
                        a = np.append(angs[i-1:],angs[0]+2 *np.pi)
                    except ValueError:
                        print self.position[i-1:].shape,self.position[:1].shape
                        raise
                    lpos = (scipy.interpolate.lagrange(a,x),scipy.interpolate.lagrange(a,y))
                    lvel = (scipy.interpolate.lagrange(a,vx),scipy.interpolate.lagrange(a,vy))
     #          print i,j,range(i-1,j+1),angs[i-1:j+1],self.position[i-1:j+1]
     #          print np.arange(angs[i-1],angs[j],(angs[j]-angs[i-1])/(2*(j+1-i)))+ (angs[j]-angs[i-1])/(2*(j+1-i))
                    for ang in np.arange(angs[i-1],angs[0]+2*np.pi,(angs[0]+2*np.pi-angs[i-1])/(2*(j+1-i)))+(angs[0]+2*np.pi-angs[i-1])/(2*(j+1-i)):
     #                  print "angles we interpolate at ",ang, "inserting"
                        pos.append([lpos[0](ang),lpos[1](ang)])
     #                  print [lpos[0](ang),lpos[1](ang)]
                        vel.append([lvel[0](ang),lvel[1](ang)])
                        try:
                            eps.append((angs[0]+np.pi*2-angs[i-1])*0.5/(j-i+1.))
                        except:
                            print j
                            raise
                i = j+2                
            else:
 #              print angs[i], self.position[i]
                pos.append(self.position[i])
                vel.append(self.velocity[i])
                eps.append(self.eps[i])
                i += 1
        self.position = np.array(pos)
        self.velocity = np.array(vel)
        self.eps = np.array(eps)

    def max_move(self):
        return self.h * max([max(np.linalg.norm(self.accel,axis=1)),max(np.linalg.norm(self.velocity,axis=1))])
        
    def increment(self):
        print len(self.position)," loops"
        h = self.h
        self.ext,self.extext = self.compute_extension()
        accE = self.acc(self.ext,self.extext)
        self.velocityEuler = self.velocity + h * self.k/self.m * accE
        self.positionEuler = self.position + h * self.velocityEuler 
        self.extEuler,self.extextEuler = self.compute_extension_Euler() 
        accH = self.acc(self.extEuler,self.extextEuler)
        self.accel = accH
#        if self.max_move()> .01:
#            self.h = 0.5 * h
#            return self.position , self.velocity
#        if self.max_move()< .003:
#            self.h = 2 * h
#            return self.position , self.velocity
        vel = self.velocity +  h * self.k/self.m * 0.5 * (accH + accE)
        pos = self.position + h*(self.velocity + vel)*0.5
        self.position = pos
        self.velocity = vel
##        self.curvature = string.max_curvature()
        self.time_elapsed += h
#        self.re_sample()
        if np.random.rand()/self.num_points**2 < .00001:
            self.faster_interaction()
        return self.position , self.velocity

    def energy(self):
        vs = np.array([np.linalg.norm(vel,axis = 1)**2 for vel in self.velocity])
        lengths = np.array([np.linalg.norm(ex,axis = 1) for ex in self.ext])
        vs = np.concatenate(vs)
        lengths = np.concatenate(lengths)
        eps = np.concatenate(self.eps)
        return 2 * np.pi *( (0.5 * self.m *eps.transpose().dot(vs)).sum() + (self.k *eps.transpose().dot(lengths)).sum())

    def interaction(self):
        out = []
        outvel = []
        num_loops = len(self.position)
        for loop_index in range(num_loops):
            loop = self.position[loop_index]
            loopvel = self.velocity[loop_index]
            loop_processed = False
            if len(loop) > 6:
                for element_index in range(1,len(loop)-1,5):
                    for other_element_index in range(max(element_index+5,element_index + len(loop)/3,5),min(len(loop)-5+element_index,len(loop))):
                        if type(self.position[0][0][0]) != np.float64:
                            print "Oh Noes!",element_index, other_element_index,
                        try:
                            if np.linalg.norm(loop[element_index] - loop[other_element_index]) <min(np.linalg.norm(loop[element_index] - loop[element_index+1]),0.1):
                                if np.random.rand() < .005 and not loop_processed:
                                    loop1 = np.concatenate((loop[other_element_index:],loop[:element_index]))
                                    loopvel1 = np.concatenate((loopvel[other_element_index:],loopvel[:element_index]))
                                    loop2 = loop[element_index:other_element_index]
                                    loopvel2 = loopvel[element_index:other_element_index]
                                    out.append(loop1)
                                    outvel.append(loopvel1) 
                                    out.append(loop2)
                                    outvel.append(loopvel2)
                                    loop_processed = True
                                    print out, outvel,"great!"
                        except ValueError:
                           # print loop_index, element_index, other_element_index#, loop[element_index]
                            raise
                        if loop_processed:
                            break
                    if loop_processed:
                        break
            if not loop_processed:
                out.append(loop)
                outvel.append(loopvel)
        self.position = np.array(out)
        self.velocity = np.array(outvel)
        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])
                            
    def fast_interaction(self):
        out = []
        outvel = []
        num_loops = len(self.position)
        for loop_index in range(num_loops):
            loop = self.position[loop_index]
            loopvel = self.velocity[loop_index]
            loop_processed = False
            if len(loop) > 6:
                element_index = np.random.randint(0,len(loop))
                for other_element_index in range(max(element_index+5,element_index + len(loop)/3,5),min(len(loop)-5+element_index,len(loop))):
                    if type(self.position[0][0][0]) != np.float64:
                        print "Oh Noes!",element_index, other_element_index,
                    try:
                        if np.linalg.norm(loop[element_index] - loop[other_element_index]) <min(np.linalg.norm(loop[element_index] - loop[element_index+1]),0.1):
                            if np.random.rand() < .0000002*len(loop)**2 and not loop_processed:
                                loop1 = np.concatenate((loop[other_element_index:],loop[:element_index]))
                                loopvel1 = np.concatenate((loopvel[other_element_index:],loopvel[:element_index]))
                                loop2 = loop[element_index:other_element_index]
                                loopvel2 = loopvel[element_index:other_element_index]
                                out.append(loop1)
                                outvel.append(loopvel1) 
                                out.append(loop2)
                                outvel.append(loopvel2)
                                loop_processed = True
                                print out, outvel,"great!"
                    except ValueError:
                       # print loop_index, element_index, other_element_index#, loop[element_index]
                        raise
                    if loop_processed:
                        break
            if not loop_processed:
                out.append(loop)
                outvel.append(loopvel)
        self.position = np.array(out)
        self.velocity = np.array(outvel)
        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])

    def faster_interaction(self):
        out = []
        outvel = []
        loop_index_1,loop_index_2 = np.random.randint(0,len(self.position)),np.random.randint(0,len(self.position))
        for loop_index,loop in enumerate(self.position):
            loopvel = self.velocity[loop_index]
            if loop_index != loop_index_1 and loop_index != loop_index_2:
                out.append(loop)
                outvel.append(loopvel)
                print "added non-interacting loops"
        if loop_index_1 == loop_index_2:
            loop = self.position[loop_index_1]
            loop_processed = False
            if len(loop) > 6:
                loopvel = self.velocity[loop_index_1]
                element_index = np.random.randint(0,len(loop))
                other_element_index = np.random.randint(0,len(loop))
                if abs(other_element_index-element_index)>max(5,len(loop)/3):#something just to make the string kinda long and to prevent too-easy self-interaction:
#                    print "Interaction possible at loop ",loop_index_1," sites ",element_index,other_element_index 
#                    print np.linalg.norm(loop[element_index] - loop[other_element_index])
                    if np.linalg.norm(loop[element_index] - loop[other_element_index]) < min(np.linalg.norm(loop[element_index] - loop[(element_index+1)%len(loop)]),0.1):
                        print "INTERACTION!"
                        loop1 = np.concatenate((loop[other_element_index:],loop[:element_index]))
                        loopvel1 = np.concatenate((loopvel[other_element_index:],loopvel[:element_index]))
                        print "1st loop ok"
                        loop2 = loop[element_index:other_element_index]
                        loopvel2 = loopvel[element_index:other_element_index]
                        print "2ns loop ok"
                        out.append(loop1)
                        outvel.append(loopvel1) 
                        out.append(loop2)
                        outvel.append(loopvel2)
                        print "loops added"
                        loop_processed = True
            if not loop_processed:
                out.append(loop)
                outvel.append(loopvel)
        else: #loops different
            out.append(self.position[loop_index_1])
            outvel.append(self.velocity[loop_index_1])
            out.append(self.position[loop_index_2])
            outvel.append(self.velocity[loop_index_2])
        self.position = np.array(out)
        self.velocity = np.array(outvel)
        self.eps = np.array([np.ones(pos.shape)*2.*np.pi / len(pos) for pos in self.position])
        if len(np.concatenate(self.position)) != self.num_points or len(np.concatenate(self.velocity)) != self.num_points:
            raise ValueError("You lost some points")

                    

    def plotter(self,N):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-2, 2), ylim=(-2, 2))
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
            other_text.set_text('h = ' + str(self.h))

            return line,  time_text, energy_text,other_text

        from time import time
        t0 = time()
        animate(0)
        t1 = time()
        interval = (100 * self.h - (t1 - t0))
        print interval

        ani = animation.FuncAnimation(fig, animate, frames=300,
                                      interval=interval, blit=True, init_func=init)

        plt.show()



poss,vels = (
    np.array([np.array([
        cos(np.arange(0,1,0.0025) * 2 * np.pi),
        sin(np.arange(0,1,0.0025) * 2 * np.pi)]).transpose()]),
    np.array([0.5*np.pi* np.array([1*sin(np.arange(0,1,0.01) * 3* 2 * np.pi),
                cos(np.arange(0,1,0.01)*2 * 2 * np.pi)]).transpose()]))
#vels = vels - vels.sum(0)/len(vels)
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


pos = np.array([
    np.array([cos(np.arange(0,1,0.01) * 2 * np.pi),
              sin(np.arange(0,1,0.01) * 2 * np.pi)]).transpose(),
    ])

ssttring = QString(init_pos = pos,hh=0.01)

