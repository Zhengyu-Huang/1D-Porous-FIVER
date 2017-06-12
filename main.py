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
    rho = 1.225
    Mach = 0.0
    p = 100000.0
    c = np.sqrt(gamma*p/rho)
    v = Mach*c
    dp = 10000.0
    '''
    v_ref = v
    p_ref = rho*v*v

    p = p/p_ref
    dp = dp/p_ref
    rho = 1
    v /= v_ref
    c /= v_ref
    '''
    fluid_info = ['constant', gamma, rho, v, p]
    #fluid_info = ['shock_tube', gamma, 5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950]
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
    #material = ['propeller_Riemann', dp]
    material = ['propeller_new_source', dp]
    structure = Structure(ms, ks, L, mode, material)

    '''
    Initialize embedded solver
        time_info: [t_end, dt], simulation end time and dt
        solver_type: 'base' or 'Dante',
                     base: 1st order FIVER(no inactive node)
                     Dante:  Dante's new version of FIVER
    '''

    t_end = 0.002
    CFL = 0.2
    dt = CFL*float(L)/float(N-1)/(v + c)
    time_info = [t_end, dt]
    #solver_type = 'Dante'
    solver_type = 'base'
    if solver_type == 'base':
        from Embedded_Explicit_Solver_Base import Embedded_Explicit_Solver_Base

        solver = Embedded_Explicit_Solver_Base(fluid, structure, time_info)


    solver._solve()
    #solver._draw()
    solver._save()
    #Exact_solution.Receding_Result(t_end,fluid.verts,fluid_info[1:])


main()






