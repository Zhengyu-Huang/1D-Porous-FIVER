__author__ = 'daniel'

import numpy as np

global SIV_ORDER
global INTERPOLATION_TOL
SIV_ORDER = 2
INTERPOLATION_TOL = 0.1

import Utility

def _flux(ww, gamma):
    u = Utility.conser_to_pri(ww, gamma)
    [gamma] = eos
    [rho, v, p] = u
    return np.array([rho*v, rho*v*v +p, (rho*v*v/2 + gamma * p/(gamma - 1))*v])




# this is Fluid Fluid Roe Flux function
def _Roe_flux(u_l, u_r, gamma):

    [rho_l, v_l, p_l] = u_l;
    [rho_r, v_r, p_r] = u_r;

    assert(p_r >0 and rho_r > 0 and rho_l >0 and p_l > 0)
    h_l = v_l*v_l/2.0 + gamma * p_l/(rho_l * (gamma - 1));
    h_r = v_r*v_r/2.0 + gamma * p_r/(rho_r * (gamma - 1));

    # compute the Roe-averaged quatities
    v_rl = (np.sqrt(rho_l)*v_l + np.sqrt(rho_r)*v_r)/ (np.sqrt(rho_r) + np.sqrt(rho_l));
    h_rl = (np.sqrt(rho_l)*h_l + np.sqrt(rho_r)*h_r)/ (np.sqrt(rho_r) + np.sqrt(rho_l));
    c_rl = np.sqrt((gamma - 1)*(h_rl - v_rl * v_rl/2));
    rho_rl = np.sqrt(rho_r * rho_l);


    du = v_r - v_l
    dp = p_r - p_l
    drho = rho_r - rho_l
    du1 = du + dp/(rho_rl * c_rl);
    du2 = drho - dp/(c_rl * c_rl);
    du3 = du - dp/(rho_rl * c_rl);

    #compute the Roe-average wave speeds
    lambda_1 = v_rl + c_rl;
    lambda_2 = v_rl;
    lambda_3 = v_rl - c_rl;

    #compute the right characteristic vectors
    r_1 =  rho_rl/(2 * c_rl) * np.array([1 ,v_rl + c_rl, h_rl + c_rl * v_rl]);
    r_2 = np.array([1, v_rl, v_rl * v_rl/2]);
    r_3 = -rho_rl/(2 * c_rl) * np.array([1,v_rl - c_rl, h_rl - c_rl * v_rl]);

    f_l = np.array([rho_l*v_l, rho_l*v_l*v_l +p_l, (rho_l*v_l*v_l/2.0 + gamma * p_l/(gamma - 1))*v_l])

    f_r = np.array([rho_r * v_r, rho_r * v_r * v_r + p_r, (rho_r * v_r * v_r / 2.0 + gamma * p_r / (gamma - 1)) * v_r])

    return 0.5*(f_l + f_r) - 0.5* (np.fabs(lambda_1)*du1*r_1 + np.fabs(lambda_2)*du2*r_2 + np.fabs(lambda_3)*du3*r_3)


def _Steger_Warming(u, W_oo, dr, gamma):
    rho, v, p = u
    c = np.sqrt(gamma * p / rho)

    Dp = np.array([max(0, v),  max(0, v + c), max(0, v - c)], dtype=float)
    Dm = np.array([min(0, v),  min(0, v + c), min(0, v - c)], dtype=float)

    h = v*v/2.0 + gamma * p / (rho * (gamma - 1));
    fp =np.array([(gamma-1)/gamma*rho*Dp[0] + rho/(2*gamma)*(Dp[1] + Dp[2]),
                  (gamma - 1) / gamma * rho * Dp[0]*v + rho/(2*gamma)*(Dp[1]*(v+c) + Dp[2]*(v-c)),
                  (gamma - 1) / gamma * rho * Dp[0] * 0.5*v**2 + rho / (2 * gamma) * (Dp[1] * (h +c*v) + Dp[2] * (h - c*v))])

    Q = np.array(    [[1,       rho/(2*c),          -rho/(2*c)],
                     [v,       rho/(2*c)*(v+c),    -rho/(2*c)*(v-c)],
                     [v**2/2,  rho/(2*c)*(h + c*v),      -rho/(2*c)*(h - c*v)]])


    Qinv = (gamma-1)/(rho*c)*np.array([[rho/c*(-v**2/2 + c**2/(gamma - 1)), rho/c*v, -rho/c],
                                    [v**2/2 - c*v/(gamma-1), -v + c/(gamma-1), 1],
                                    [-v**2/2 - c*v/(gamma-1), v + c/(gamma-1), -1]])

    fm = np.dot(Q, Dm * np.dot(Qinv, W_oo))
    #print('identity is ', np.dot(Q,Qinv))

    return dr*(fp + fm)







#This is 1-d case
# w_l is the fluid state primitive variable
# v is the piston or wall speed
# fluid_normal is the normal of the fluid interface
# if fluid_normal = 1 piston is on the right
# if fluid_normal =-1 piston is on the left
def _FSRiemann(u_l, v, fluid_normal, gamma):
    [rho_l, v_l, p_l] = u_l;
    #left facing shock case
#    if(fluid_normal < 0 and np.fabs(v_l) > 0.01):
#        a = 1
    if(v_l*fluid_normal > v*fluid_normal):
        #print "Shock Wave FSRiemann"
        a = 2/((gamma + 1)*rho_l);
        b = p_l*(gamma - 1)/(gamma + 1)
        phi = a/(v - v_l)**2

        p = p_l + (1 + np.sqrt(4*phi*(p_l + b) + 1))/(2*phi)
        rho = rho_l*(p/p_l + (gamma - 1)/(gamma + 1))/(p/p_l * (gamma - 1)/(gamma + 1) + 1)
    #left facing rarefactions case
    else:
        #print "Rarefactions FSRiemann"
        c_l = np.sqrt(gamma*p_l/rho_l);
        p = p_l*(-(gamma - 1)/(2*c_l)*(v - v_l)*fluid_normal + 1)**(2*gamma/(gamma - 1))
        rho = rho_l*(p/p_l)**(1/gamma);
    u_s = np.array([rho, v, p]);
    return u_s



'''
# this is Fluid Fluid Roe Flux function
def _Roe_flux(u_l, u_r, gamma):

    [rho_l, v_l, p_l] = u_l;
    [rho_r, v_r, p_r] = u_r;

    assert(p_r >0 and rho_r > 0 and rho_l >0 and p_l > 0)
    c_l = np.sqrt(gamma * p_l/ rho_l)
    c_r = np.sqrt(gamma * p_r/ rho_r)
    w_l = np.array([rho_l, rho_l*v_l, rho_l*v_l*v_l/2 + p_l/(gamma - 1)])
    w_r = np.array([rho_r, rho_r*v_r, rho_r*v_r*v_r/2 + p_r/(gamma - 1)])
    h_l = v_l*v_l/2.0 + gamma * p_l/(rho_l * (gamma - 1));
    h_r = v_r*v_r/2.0 + gamma * p_r/(rho_r * (gamma - 1));

    # compute the Roe-averaged quatities
    v_rl = (np.sqrt(rho_l)*v_l + np.sqrt(rho_r)*v_r)/ (np.sqrt(rho_r) + np.sqrt(rho_l));
    h_rl = (np.sqrt(rho_l)*h_l + np.sqrt(rho_r)*h_r)/ (np.sqrt(rho_r) + np.sqrt(rho_l));
    c_rl = np.sqrt((gamma - 1)*(h_rl - v_rl * v_rl/2));
    rho_rl = np.sqrt(rho_r * rho_l);


    du = v_r - v_l
    dp = p_r - p_l
    drho = rho_r - rho_l
    du1 = du + dp/(rho_rl * c_rl);
    du2 = drho - dp/(c_rl * c_rl);
    du3 = du - dp/(rho_rl * c_rl);

    #compute the Roe-average wave speeds
    lambda_1 = v_rl + c_rl;
    lambda_2 = v_rl;
    lambda_3 = v_rl - c_rl;

    #compute the right characteristic vectors
    r_1 =  rho_rl/(2 * c_rl) * np.array([1 ,v_rl + c_rl, h_rl + c_rl * v_rl]);
    r_2 = np.array([1, v_rl, v_rl * v_rl/2]);
    r_3 = -rho_rl/(2 * c_rl) * np.array([1,v_rl - c_rl, h_rl - c_rl * v_rl]);

    Q = np.array([[1 ,                   1,                       1],
                 [v_rl - c_rl,         v_rl,                (v_rl + c_rl)],
                 [h_rl - c_rl*v_rl,   v_rl*v_rl/2,        (h_rl + c_rl * v_rl)]])

    D = np.diag([np.fabs(v_rl - c_rl), np.fabs(v_rl), np.fabs(v_rl + c_rl)])

    Qinv = np.array([[v_rl/(4*c_rl)*(2 + (gamma-1)*v_rl/c_rl) , -(1 + (gamma-1)*v_rl/c_rl)/(2*c_rl)    ,(gamma-1)/(2*c_rl*c_rl)],
                    [1 - (gamma-1)*v_rl*v_rl/(2*c_rl*c_rl),         (gamma-1)*v_rl/(c_rl**2),             -(gamma-1)/(c_rl**2)],
                    [-v_rl/(4*c_rl)*(2 - (gamma-1)*v_rl/c_rl),  (1 - (gamma-1)*v_rl/c_rl)/(2*c_rl)    ,(gamma-1)/(2*c_rl*c_rl)]])

    #compute the fluxes
    mach = v_rl/ c_rl
    if(mach <=-1):
        flux = np.array([rho_r*v_r, rho_r*v_r*v_r + p_r, rho_r*v_r*h_r])
    elif(mach <= 0 and mach >= -1):
        flux = np.array([rho_r * v_r, rho_r*v_r**2 + p_r,rho_r*h_r*v_r]) - r_1*lambda_1*du1;
    elif(mach >= 0 and mach <= 1):
        flux = np.array([rho_l * v_l, rho_l*v_l**2 + p_l,rho_l*h_l*v_l]) + r_3*lambda_3*du3;
    else:
        flux = np.array([rho_l * v_l, rho_l*v_l**2 + p_l,rho_l*h_l*v_l])

    return flux
'''