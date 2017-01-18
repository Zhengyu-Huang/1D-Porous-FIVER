__author__ = 'daniel'

import numpy as np


def pri_to_conser(fluid,gamma):
    [rho, v, p] = fluid;
    return  np.array([rho, rho*v, rho*v*v/2 + p/(gamma - 1)])


def conser_to_pri(fluid, gamma):
    [w1, w2, w3] = fluid
    rho = w1;
    v = w2/w1;
    p = (w3 - w2*v/2) * (gamma - 1)
    return np.array([rho, v, p])

def pri_to_conser_all(V,W,gamma):
    W[:,0] = V[:,0]
    W[:,1] = V[:,1]*V[:,0]
    W[:,2] = 0.5*V[:,0]*V[:,1]**2 + V[:,2]/(gamma - 1.0)

    #change conservative variables to primitive variables
def conser_to_pri_all(W, V,gamma):
    V[:,0] = W[:,0]
    V[:,1] = W[:,1]/W[:,0]
    V[:,2] = (W[:,2] - 0.5*W[:,1]*V[:,1]) * (gamma - 1.0)


def safe_interpolation(w1, x1, w2, x2, x3, gamma, check):
    [dx,tol] = check
    u1 = conser_to_pri(w1, gamma)
    u2 = conser_to_pri(w2, gamma)

    if(abs(x2 - x1) > tol*dx):
        u3 = (u2 - u1) * (x3 - x2)/(x2 - x1) + u2
        if(u3[0] >= 0 and u3[2] >= 0):
            w3 =pri_to_conser(u3,gamma)
            return w3
        else:
            print("w1 is ", w1, " w2 is ", w2)
            print("In Interpolation, encounter negative values ")
    else:
        print("In Interpolation, points are too closed")

    if(abs(x1 - x3) > abs(x2 -x3)):
        return w2
    else:
        return w1

def interpolation(w1, x1, w2, x2, x3, gamma):

    u1 = conser_to_pri(w1, gamma)
    u2 = conser_to_pri(w2, gamma)


    u3 = (u2 - u1) * (x3 - x2)/(x2 - x1) + u2

    w3 =pri_to_conser(u3,gamma)
    return w3


def pressure_load_interpolation(w1, x1, w2, x2, x3, gamma):
    u1 = conser_to_pri(w1, gamma)
    u2 = conser_to_pri(w2, gamma)

    p1,p2 = u1[2],u2[2]

    p3 = (p2 - p1) * (x3 - x2) / (x2 - x1) + p2
    return p3


