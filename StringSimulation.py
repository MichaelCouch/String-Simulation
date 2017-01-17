import numpy as np
import math
import pandas as pd
num_steps  = 100 # number of time intervals
num_angles = 100 # elements in S^1
mass_per_unit_angle_div_springyness = 1
total_time = 10.
h = total_time / num_steps
eps = 2. * math.pi / num_angles
planck_length = 0.01

class QString:

    def __init__(self,
                 init_pos = np.zeros([num_angles,3])
                 init_vel = np.zeros([num_angles,3])
                 m = 1.,
                 k = 1.):
        self.position = init_pos[0]
        self.velocity = init_vel[1]
        self.params = (m,k)
        self.time_elapsed = 0.
        self.eps = 2.*np.pi / len(self.position)
        if self.position.shape != self.velocity.shape:
            raise 
        
    def rotate(self,attr,n=1):
        return self.attr[n:] + self.attr[:n]
    
    def increment(self,dt):
        pos = self.position + h * self.velocity
        posdd = (self.rotate(position,-1) - 2 * self.position + self.rotate(position,1)) / (self.eps**2)
        posd  = (self.rotate(position,1) - self.position)/self.eps
        len_posd = np.linalg.norm(posd,axis=1)
        len_posd = np.array(map(lambda a: max(a,planck_length),len_posd))
        acc = np.array(map(lambda dd,d,de: dd/de - dd.dot(d) * d / (de**3),posdd,posd,len_posd))
        vel = self.velocity + h * k / m * acc
        self.position = pos
        self.velocity = vel
        return self.position , self.velocity

    def energy(self):
        vs = np.linalg.norm(self.velocity,axis = 1)**2
        lengths = np.linalg.norm,self.position,axis = 1
        return 2 * np.pi * self.eps*( 0.5 * m * vs.sum() + k * lengths.sum()) 
