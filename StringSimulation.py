from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
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
        self.eps = np.ones(self.position.shape)*2.*np.pi / len(self.position)
        if self.position.shape != self.velocity.shape:
            raise np.linalg.LinAlgError("mismatched position and velocity input shape")
        
    def rotate(self,n=1):
        return np.concatenate((self.position[n:],self.position[:n])), np.concatenate((self.velocity[n:], self.velocity[:n]))
    
    def increment(self,h):
        orig_pos,orig_vel = self.position,self.velocity
        posddE = (self.rotate(-1)[0] - 2 * self.position + self.rotate(1)[0]) / (self.eps**2)
        posdE  = (self.rotate(1)[0] - self.position)/self.eps
        len_posdE = np.linalg.norm(posdE,axis=1)
        len_posdE = np.array(map(lambda a: max(a,planck_length),len_posdE))
        accE = np.array(map(lambda dd,d,de: dd/de - dd.dot(d) * d / (de**3),posddE,posdE,len_posdE))
        velE = self.velocity + h * self.k/self.m * accE
        posE = self.position + h * velE
        self.position = posE
        self.velocity = velE
        
        posdd = (self.rotate(-1)[0] - 2 * self.position + self.rotate(1)[0]) / (self.eps**2)
        posd  = (self.rotate(1)[0] - self.position)/self.eps
        len_posd = np.linalg.norm(posd,axis=1)
        len_posd = np.array(map(lambda a: max(a,planck_length),len_posd))
        acc = np.array(map(lambda dd,d,de: dd/de - dd.dot(d) * d / (de**3),posdd,posd,len_posd))
        vel = orig_vel + h * self.k/self.m * 0.5 * (acc + accE)
        pos = orig_pos + h * (orig_vel + vel)*0.5
        self.position = pos
        self.velocity = vel
        
        string.time_elapsed += h
        return self.position , self.velocity

    def energy(self):
        vs = np.linalg.norm(self.velocity,axis = 1)**2
        lengths = np.linalg.norm(self.position,axis = 1)
        return 2 * np.pi *( 0.5 * self.m *(self.eps.transpose()[0]).dot(vs).sum() + self.k *(self.eps.transpose()[0]).dot(lengths).sum())

pos = np.array([cos(np.arange(0,1,0.01) * 2 * np.pi),
                sin(np.arange(0,1,0.01) * 2 * np.pi)])
pos =  pos.transpose()
vel = 0.375 * np.pi* np.array([-1*sin(np.arange(0,1,0.01) * 2 * np.pi),
                np.arange(0,1,0.01) * 0])
vel = vel.transpose()
string = QString(init_pos = pos, init_vel = vel)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
print line

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
