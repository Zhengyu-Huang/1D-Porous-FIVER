__author__ = 'daniel'
#################################
#The body motion parameters can be set in body_motion.py
#Geometry, error check parameters can be set in global_variable.py
#Test, fluid parameter could be set here
#The initial condition could be set in FIVER.py, we have constant and linear initialization
######################################

from Fluid import Fluid
from Structure import Structure
from Embedded_Explicit_Solver_Base import Embedded_Explicit_Solver_Base
import Exact_solution

def main():



    #TODO we assume that Time divides by dt


    #arg[0] is gamma
    #arg[1:4] is desity, velocity and pressure
    #in FIVER, there is an initalization function
    #to initialize the fluid state,
    #fluid_info = [gamma, density, velocity, pressure]
    gamma = 1.4
    rho = 1.225
    v = 0
    p = 1.0
    fluid_info = ['constant', gamma, rho, v, p]


    #fluid_info
    # number of fluid domain vertices
    N = 1000

    # simulation domain [0,L]
    L = 4.0
    bc = ['far_field', 'far_field']
    fluid= Fluid(L,N,fluid_info,bc)


    #structure_info
    ms = 1.0
    ks = 1.0
    alpha = 0
    mode = 1
    structure = Structure(ms, ks, alpha, L, mode)


    #Embedded explicit solver
    # simulation time
    t_end = 1.0
    CFL = 0.5
    dt = CFL*float(L)/float(N-1)
    time_info = [t_end, dt]
    solver = Embedded_Explicit_Solver_Base(fluid, structure, time_info)

    solver._solve()
    solver._draw()
    Exact_solution.Receding_Result(t_end,fluid.verts,fluid_info[1:])


main()






