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
            self.position = np.zeros(self.velocity.shape)
        if self.velocity == None:
            self.velocity = np.zeros(self.position.shape)
        self.m = m
        self.k = k
        self.time_elapsed = 0.
        self.eps = np.ones(len(self.position))*2.*np.pi / len(self.position)
        if self.position.shape != self.velocity.shape:
            raise np.linalg.LinAlgError("mismatched position and velocity input shape")
        self.ext,self.extext = self.compute_extension()
        
    def rotate(self,n=1):
        return np.concatenate((self.position[n:],self.position[:n])), np.concatenate((self.velocity[n:], self.velocity[:n])),np.concatenate((self.eps[n:], self.eps[:n]))

    def rotate_Euler(self,n=1):
        return np.concatenate((self.positionEuler[n:],self.positionEuler[:n])),np.concatenate((self.eps[n:], self.eps[:n]))

#    def posdd(self):
#        return np.array(
#            map(lambda a,b,c: a/(b*c),
#                (self.rotate(-1)[0] - 2 * self.position + self.rotate(1)[0]),
#                self.eps,
#                self.rotate(-1)[2])
#                )

    def compute_extension(self):
        yminus, vminus, epsminus=self.rotate(-1)  
        y, eps = self.position, np.array([self.eps,self.eps]).transpose()  
        yplus, vplus, epsplu=self.rotate(1)
        epsplus = np.array([epsplu,epsplu]).transpose()   
        return  ( 1/(eps + epsplus) * ((yplus - y)*eps/(epsplus) + (y - yminus)*epsplus/eps),
                    2/(eps + epsplus) * ( (yplus - y)/epsplus - (y - yminus)/eps ) )

    def compute_extension_Euler(self):
        yminus, epsminus=self.rotate_Euler(-1)  
        y, eps = self.positionEuler, np.array([self.eps,self.eps]).transpose()  
        yplus, epsplu=self.rotate_Euler(1)
        epsplus = np.array([epsplu,epsplu]).transpose()  
        return  ( 1/(eps + epsplus) * ((yplus - y)*eps/(epsplus) + (y - yminus)*epsplus/eps),
                    2/(eps + epsplus) * ( (yplus - y)/epsplus - (y - yminus)/eps ) )        
        
    def acc(self,ext,extext):
        acc = np.array(
            map(lambda dd,d,de: dd, #dd/de - dd.dot(d) * d / (de**3) ,
                extext,    ext,   np.linalg.norm(ext,axis=1)))
        return acc

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
        return self.position , self.velocity

    def energy(self):
        vs = np.linalg.norm(self.velocity,axis = 1)**2
        lengths = np.linalg.norm(self.position,axis = 1)
        return 2 * np.pi *( 0.5 * self.m *(self.eps).dot(vs).sum() + self.k *(self.eps).dot(lengths).sum())

#    def max_curvature(self):
#        return max(np.linalg.norm(np.array(map(lambda a,b: a*b, self.acc,self.eps)),axis = 1))

    def plotter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-2, 2), ylim=(-2, 2))
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        other_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)

        def init():
            """initialize animation"""
            line.set_data([], [])
            time_text.set_text('')
            energy_text.set_text('') 
            return line, time_text, energy_text

        def animate(i):
            """perform animation step"""
            self.increment()
            x,y = self.position.transpose()
            line.set_data(x,y)
        #    time_text.set_text('curve = ' + str(string.curvature))
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



poss,vels = np.array([cos(np.arange(0,1,0.01) * 2 * np.pi),
                sin(np.arange(0,1,0.01) * 2 * np.pi)]).transpose(),0.5*np.pi* np.array([1*sin(np.arange(0,1,0.01) * 3* 2 * np.pi),
                cos(np.arange(0,1,0.01)*2 * 2 * np.pi)]).transpose()
vels = vels - vels.sum(0)/len(vels)
string = QString(init_pos = poss,init_vel = vels)
    
ttring = QString(init_pos = poss,init_vel = 0*vels)

pos = 5* np.array(
    [np.cos(np.array([np.arange(10)]).transpose().dot(np.array([np.arange(0,1,0.01)]))).transpose().dot(np.random.rand(10)/10-.5/10),
     np.cos(np.array([np.arange(10)]).transpose().dot(np.array([np.arange(0,1,0.01)]))).transpose().dot(np.random.rand(10)/10-.5/10)]).transpose()+poss
vel = np.array(
    [np.cos(np.array([np.arange(10)]).transpose().dot(np.array([np.arange(0,1,0.01)]))).transpose().dot(np.random.rand(10)/10-.5/10),
     np.sin(np.array([np.arange(10)]).transpose().dot(np.array([np.arange(0,1,0.01)]))).transpose().dot(np.random.rand(10)/10-.5/10)]).transpose()+vels
vel = vel - vel.sum(0)/len(vels)
sstring = QString(init_pos = pos,init_vel = vel)
