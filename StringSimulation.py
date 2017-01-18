from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.animation as animation


num_steps  = 10000 # number of time intervals
total_time = 10.
h = total_time / num_steps
planck_length = 0.01

class QString:

    def __init__(self,
                 init_pos = None,
                 init_vel = None,
                 m = 1.,
                 k = 1.):
        self.position = init_pos
        self.velocity = init_vel
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
        
    def rotate(self,n=1):
        return np.concatenate((self.position[n:],self.position[:n])), np.concatenate((self.velocity[n:], self.velocity[:n]))

    def posdd(self):
        return np.array(
            map(lambda a,b: a/b,
                (self.rotate(-1)[0] - 2 * self.position + self.rotate(1)[0]),
                self.eps**2)
            )

    def posd(self):
        return np.array(
            map(lambda a,b: a/b,
                (self.position - self.rotate(1)[0]),
                self.eps))

    def re_sample(self):
        threshold = 1.
        curvatures = numpy.linalg.norm(
                    np.array(map(lambda a,b: a*b, self.acc,self.eps)),
                    axis=1)
        pos = []
        vel = []
        eps = []
        times = np.cumsum(self.eps)
        
        i = 0
        while i < len(curvatures):
            if curvatures[i]> threshold:
                j = i
                while curvatures[j] > threshold and j < len(self.position):
                    j += 1
                x,y = self.position[i-1,j+1].transpose()
                vx,vy = self.velocity[i-1,j+1].transpose()
                t = times[i-1,j+1]
                lpos = np.array([scipy.interpolate.lagrange(t,x),scipy.interpolate.lagrange(t,y)])
                lvel = np.array([scipy.interpolate.lagrange(t,vx),scipy.interpolate.lagrange(t,vy)])
                for time in np.append(np.arange(times[i,j+1],(times[j+1]-times[i-1])*0.5),times[j+1]):
                    pos.append([lpos[0](time),lpos[1](time)])                                        
                    vel.append([lvel[0](time),lvel[1](time)])
                    eps.append((times[j+1]-times[i-1])*0.5/(j-i+1.))
                i = j+1
            else:
                pos.append(self.position[i])
                vel.append(self.velocity[i])
                eps.append(self.eps[i])
            i += 1
        self.position = np.array(pos)
        self.velocitity = np.array(vel)
        
    def increment(self,h):
        orig_pos,orig_vel = self.position,self.velocity
        posddE = self.posdd()
        posdE  = self.posd()
        len_posdE = np.linalg.norm(posdE,axis=1)
        len_posdE = np.array(map(lambda a: max(a,planck_length),len_posdE))
        accE = np.array(map(lambda dd,d,de: dd/de - dd.dot(d) * d / (de**3),posddE,posdE,len_posdE))
        velE = self.velocity + h * self.k/self.m * accE
        posE = self.position + h * velE
        self.position = posE
        self.velocity = velE
        
        posddH = self.posdd()
        self.posddH = posddH
        posdH  = self.posd()
        len_posdH = np.linalg.norm(posdH,axis=1)
        len_posdH = np.array(map(lambda a: max(a,planck_length),len_posdH))
        acc = np.array(map(lambda dd,d,de: dd/de - dd.dot(d) * d / (de**3),posddH,posdH,len_posdH))
        string.acc = acc
        vel = orig_vel + h * self.k/self.m * 0.5 * (acc + accE)
        pos = orig_pos + h * (orig_vel + vel)*0.5
        self.position = pos
        self.velocity = vel
        
        string.curvature = string.max_curvature()
        string.time_elapsed += h
        return self.position , self.velocity

    def energy(self):
        vs = np.linalg.norm(self.velocity,axis = 1)**2
        lengths = np.linalg.norm(self.position,axis = 1)
        return 2 * np.pi *( 0.5 * self.m *(self.eps).dot(vs).sum() + self.k *(self.eps).dot(lengths).sum())

    def max_curvature(self):
        return max(np.linalg.norm(np.array(map(lambda a,b: a*b, self.acc,self.eps)),axis = 1))

pos = np.array([cos(np.arange(0,1,0.01) * 2 * np.pi),
                sin(np.arange(0,1,0.01) * 2 * np.pi)])
pos =  pos.transpose()
#vel = 0.375 * np.pi* np.array([-1*sin(np.arange(0,1,0.01) * 2 * np.pi),
#                np.arange(0,1,0.01) * 0])
#vel = vel.transpose()
string = QString(init_pos = pos)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('') 
    return line, time_text, energy_text

def animate(i):
    """perform animation step"""
    global string, h
    string.increment(h)
    x,y = string.position.transpose()
    line.set_data(x,y)
    time_text.set_text('time = ' + str(string.time_elapsed))
    energy_text.set_text('energy = %.3f J' % string.energy())

    return line,  time_text, energy_text

from time import time
t0 = time()
animate(0)
t1 = time()
interval = (100 * h - (t1 - t0))
print interval

ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=interval, blit=True, init_func=init)

plt.show()
