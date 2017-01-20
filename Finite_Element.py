import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.animation as animation


class cylinder_solver:
    """
    A relaxation-solver for a PDE with Lagranian Dirichlet bcs
    A solver for the world sheet of a string with initial and final conditions specified. 
    """
    def __init__(self,dt,dx,m=1,k=1,direction = max,
        mesh = None,
        init_guess=None,
        boundary_conditions=None,
        perturbation_order = 0):
        self.state = init_guess
        self.boundary_conditions = boundary_conditions
        self.perturbation_order = perturbation_order
        self.dt = dt
        self.dx = dx
        self.m=m
        self.k = k
        self.direction = direction
        self.increment_count = 0
        if init_guess:
            if boundary_conditions is not None:
                if init_guess[0] != boundary_conditions[0] or init_guess[-1] != boundary_conditions[1]:
                    raise ValueError("Boundary conditions do not match with initial guess")
            if mesh is not None:
                if init_guess.shape[:2] != mesh:
                    raise ValueError("mesh size does not match with that of initial guess")
            self.boundary_conditions = (self.state[0],self.state[-1])
        else:
            if boundary_conditions is not None:
                if not mesh:
                    mesh = (10,boundary_conditions.shape[1])
                if boundary_conditions.shape[1] == mesh[1]:
                    #print "interpolating"
                    self.state = self.interpolate_BCs(mesh)#interpolate between the boundary conditions to give initial state
                else:
                    raise ValueError("boundary conditions not compatible with mesh" +str(boundary_conditions.shape[1]) + str(mesh[1]))
            else:
                raise ValueError("Please provide boundary conditions")

    def interpolate_BCs(self,mesh):
        stateTrans = np.zeros((mesh[1],mesh[0],2))
        for j in range(mesh[1]):
            stateTrans[j] = np.array([np.linspace(self.boundary_conditions[0,j,0],self.boundary_conditions[1,j,0],mesh[0]),
                             np.linspace(self.boundary_conditions[0,j,1],self.boundary_conditions[1,j,1],mesh[0])]).transpose()
        return stateTrans.transpose((1,0,2))

    #def rotate(self,n=1):
    #    return (
    #        concatenate((self.state[n:]),self.state[:n]),
            
    
    def action(self): #I think we want to minimize this
        action = 0
        for i in range(self.state.shape[0]-1):
            for j in range(self.state.shape[1]):
                action +=self.lagrangian(i,j)* self.dx * self.dt
        return action
    
    def lagrangian(self,i,j):
        return 1./2. * self.m  * (self.velocity(i,j)**2).sum() + self.k *(np.linalg.norm(self.extension(i,j))**2).sum()


    def velocity(self,i,j):  
        inc_j = (j+1) % (self.state.shape[1])
        return 1./(2 * self.dt) * (self.state[i+1,j] - self.state[i,j] + self.state[i+1,inc_j]- self.state[i,inc_j])
    
    def extension(self,i,j):
        inc_j = (j+1) % (self.state.shape[1])
        return 1./(2 * self.dx) * (self.state[i+1,inc_j] - self.state[i+1,j] + self.state[i,inc_j] - self.state[i,j])
        
    def increment(self):
        en = self.action()
        decrease_order = True
        for i in range(1,self.state.shape[0] - 1):
            for j in range(self.state.shape[1]):
                inc_j = (j+1) % self.state.shape[1]
                for k in [0,1]:
                    zero = self.lagrangian(i,j) + self.lagrangian(i-1,j) + self.lagrangian(i,j-1) + self.lagrangian(i-1,j-1)
                    self.state[i,j,k] += 2**(-self.perturbation_order)
                    plus = self.lagrangian(i,j) + self.lagrangian(i-1,j) + self.lagrangian(i,j-1) + self.lagrangian(i-1,j-1)
                    self.state[i,j,k] -= 2*(2**(-self.perturbation_order))
                    minus = self.lagrangian(i,j) + self.lagrangian(i-1,j) + self.lagrangian(i,j-1) + self.lagrangian(i-1,j-1)
                    move = [minus,zero,plus].index(max([minus,zero,plus]))
                    self.state[i,j,k] += move*(2**(-self.perturbation_order))
                    if move != 1:
                        decrease_order = False
        if decrease_order:
            self.perturbation_order += 1
        self.increment_count += 1
        if (self.increment_count) % 50 ==0:
            self.perturbation_order -= 1
        return en-self.action()

    def plotter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                 xlim=(-2, 2), ylim=(-2, 2))
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
#time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
#energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        def init():
            """initialize animation"""
            line.set_data([], [])
            #time_text.set_text('')
            #energy_text.set_text('') 
            return line,# time_text, energy_text

        def animate(i):
            """perform animation step""" 
            x,y = self.state.transpose((0,2,1))[i%len(self.state)]
            line.set_data(x,y)
        #    time_text.set_text('curve = ' + str(string.curvature))
        #    time_text.set_text('time = ' + str(string.time_elapsed))
        #    energy_text.set_text('energy = %.3f J' % string.energy())
            return line,#  time_text, energy_text

        interval = 100

        ani = animation.FuncAnimation(fig, animate, frames=300,
                          interval=interval, blit=True, init_func=init)

        plt.show()
