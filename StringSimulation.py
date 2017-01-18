from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

num_steps  = 100 # number of time intervals
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
        self.eps = 2.*np.pi / len(self.position)
        if self.position.shape != self.velocity.shape:
            raise np.linalg.LinAlgError("mismatched position and velocity input shape")
        
    def rotate(self,n=1):
        return np.concatenate((self.position[n:],self.position[:n])), np.concatenate((self.velocity[n:], self.velocity[:n]))
    
    def increment(self,dt):
        pos = self.position + h * self.velocity
        posdd = (self.rotate(-1)[0] - 2 * self.position + self.rotate(1)[0]) / (self.eps**2)
        posd  = (self.rotate(1)[0] - self.position)/self.eps
        len_posd = np.linalg.norm(posd,axis=1)
        len_posd = np.array(map(lambda a: max(a,planck_length),len_posd))
        acc = np.array(map(lambda dd,d,de: dd/de - dd.dot(d) * d / (de**3),posdd,posd,len_posd))
        vel = self.velocity + h * self.k/self.m * acc
        self.position = pos
        self.velocity = vel
        return self.position , self.velocity

    def energy(self):
        vs = np.linalg.norm(self.velocity,axis = 1)**2
        lengths = np.linalg.norm(self.position,axis = 1)
        return 2 * np.pi * self.eps*( 0.5 * self.m * vs.sum() + self.k * lengths.sum())

pos = np.array([cos(np.arange(0,1,0.01) * 2 * np.pi),sin(np.arange(0,1,0.01) * 2 * np.pi)])
pos = pos.transpose()
string = QString(init_pos = pos)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
line = ax.plot([], [], 'o-', lw=2)

def init():
    """initialize animation"""
    line.set_data([], [])
#    time_text.set_text('')
#    energy_text.set_text('') 
    return line#, time_text, energy_text

def animate(i):
    """perform animation step"""
    global string, h
    string.increment(h)
    line.set_data(string.position.transpose()[0],string.position.transpose()[1])
#    time_text.set_text('time = %.1f' % pendulum.time_elapsed)
#    energy_text.set_text('energy = %.3f J' % pendulum.energy())
    return line#, time_text, energy_text

from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * h - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=interval, blit=True, init_func=init)
