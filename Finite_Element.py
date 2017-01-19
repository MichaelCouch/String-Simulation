import numpy as np


class cylinder_solver:
    """
    A relaxation-solver for a PDE with Lagranian Dirichlet bcs
    A solver for the world sheet of a string with initial and final conditions specified. 
    """
    def __init__(self,dt,dx
        mesh = None,
        init_guess=None
        boundary_conditions=None):
        if init_guess:
            if boundary_conditions:
                if init_guess[0] != boundary_conditions[0] or init_guess[-1] != boundary_conditions[1]:
                    raise ValueError("Boundary conditions do not match with initial guess")
            if mesh:
                if init_guess.shape[:2] != mesh:
                    raise ValueError("mesh size does not match with that of initial guess")
            self.state = init_guess
            self.boundary_conditions = (self.state[0],self.state[-1])
        else:
            if boundary_conditions:
                if not mesh:
                    mesh = (10,boundary_conditions.shape[1])
                if boundary_conditions.shape[1] = mesh[1]:
                    self.state = self.interpolate_BCs(mesh)#interpolate between the boundary conditions to give initial state
                else:
                    raise ValueError("boundary conditions not compatible with agree with mesh")
            else:
                raise ValueError("Please provide boundary conditions")
            
                
                
            
            
        
    
