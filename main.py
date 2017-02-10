__author__ = 'daniel'
#################################
#The body motion parameters can be set in body_motion.py
#Geometry, error check parameters can be set in global_variable.py
#Test, fluid parameter could be set here
#The initial condition could be set in FIVER.py, we have constant and linear initialization
######################################
import numpy as np
from Fluid import Fluid
from Structure import Structure
import Exact_solution

def main():


    '''
    Initialize fluid class
    fluid_info:
        constant initial condition: ['constant', gamma, density, velocity, pressure]
        shock tube initial condition: ['shock_tube', gamma, rho_l, v_l, p_l, rho_r, v_r, p_r]
        smooth initial condition: ['smooth']

    N: number of fluid domain vertices

    L: simulation domain [0,L]

    bc: boundary condtion, there are two types of boundary condition, wall or far_field
    '''
    gamma = 1.4
    rho = 1.4
    Mach = -2.0
    p = 1.0
    v = Mach*np.sqrt(gamma*p/rho)
    fluid_info = ['constant', gamma, rho, v, p]
    #fluid_info = ['shock_tube', gamma, rho, 0.0, 2*p,  rho, 0.0, p]
    N = 500
    L = 4.0
    bc = ['far_field', 'far_field']
    fluid= Fluid(L,N,fluid_info,bc)

    '''
    Initialize structure class
        ms: structure mass
        ks: structure stiffness, if the structure is modeled as piston-spring system  $$ms \dot\dot{xs} + ks(xs - xs_0) = f(t) = p_B$$
        mode: structure motion,       0: piston-spring system ;
                                      1: forced motion, advancing type 1
                                      2: forced motion, advancing type 2
                                      3: forced motion, harmonic motion
                                      4: fixed at L/2
        material: ['porous', porous_ratio] or ['propeller', dp], it can be porous material, or propeller
        L: tube length, the fluid domain is [0,L]
    '''
    ms = 1.0
    ks = 1.0
    mode = 4
    #material = ['porous', 0.1] #0 means cannot go through
    material = ['propeller_Riemann', 0.2]
    #material = ['propeller', 0.2]
    structure = Structure(ms, ks, L, mode, material)

    '''
    Initialize embedded solver
        time_info: [t_end, dt], simulation end time and dt
        solver_type: 'base' or 'Dante',
                     base: 1st order FIVER(no inactive node)
                     Dante:  Dante's new version of FIVER
    '''

    t_end = 0.5
    CFL = 0.2
    dt = CFL*float(L)/float(N-1)
    time_info = [t_end, dt]
    #solver_type = 'Dante'
    solver_type = 'base'
    if solver_type == 'base':
        from Embedded_Explicit_Solver_Base import Embedded_Explicit_Solver_Base

        solver = Embedded_Explicit_Solver_Base(fluid, structure, time_info)
    elif solver_type == 'Dante':
        from Embedded_Explicit_Solver_Dante import Embedded_Explicit_Solver_Dante

        solver = Embedded_Explicit_Solver_Dante(fluid, structure, time_info)


    solver._solve()
    solver._draw()
    #Exact_solution.Receding_Result(t_end,fluid.verts,fluid_info[1:])


main()






