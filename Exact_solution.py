import numpy as np
import matplotlib.pyplot as plt

def Structure_Acceleration(t):
    return 0


def Structure(t):
    L =4.0

    xs = L/2 + 1.0/8.0 * t**4
    vs = 1.0/2.0 * t**3

    return np.array([xs,vs])



def Exact_Solution_Point(t, x, L , fluid_info):

    [gamma, rho_l, v_l, p_l] = fluid_info




    a0 = np.sqrt(gamma*p_l/rho_l)

    [x0, v0] = Structure(0.0)

    fan_start = float(L)/2.0 - t*a0

    fan_end   = float(L)/2.0 + t*(v0*(gamma + 1)/2 - a0)

    [xs, vs] = Structure(t)

    rho = 0.0
    v   = 0.0
    if(x < fan_start):
        rho = rho_l
        v   = v_l
    elif(fan_start <= x and x <= fan_end):
        K = np.sqrt(gamma*p_l/rho_l**gamma)

        rho = a0 - (gamma - 1)*0.5*(x - L/2.0)/t
        rho = 2*rho/(K*(gamma + 1))
        rho = rho**(2/(gamma - 1))

        v   = 2.0*((x - L/2.0)/t + a0)/(gamma + 1)

    elif(fan_end <= x and x <= xs):

        tau = 0.0
        i = 0
        for i in range(100):
            tau = tau - f_fan(tau, x, t, a0, gamma)/df_fan(tau, x, t, a0, gamma)
            if(np.fabs(f_fan(tau, x, t, a0, gamma)) < 1.0e-15 ):
                break
        if(i >= 99):
            print('Divergence Newtons method')
        if(tau > t):
            print('Newtons method find another solution, change method')
            l = 0.0
            r = t
            while(1):
                tau = (l + r)/2.0
                f = f_fan(tau, x, t, a0, gamma)
                if(np.fabs(f) < 1.0e-15 ):
                    break
                elif(f > 0):
                    r = tau
                else:
                    l = tau




        [xs, vs] = Structure(tau)
        v = vs
        rho = (a0 - 0.5*(gamma - 1)*vs)/ a0

        rho = rho_l * rho**(2.0/(gamma - 1.0))

    else:
        rho = 0.0
        v   = 0.0

    p = p_l*(rho/rho_l)**gamma



    return np.array([rho, v, p])

def f_fan(tau, x, t , a0, gamma):
    [xs, vs] = Structure(tau)
    f = (0.5*(gamma + 1)*vs - a0)*(t - tau) + xs - x
    return f


def df_fan(tau, x, t, a0, gamma):
    [xs, vs] = Structure(tau)
    a_s = Structure_Acceleration(tau)
    df = -(vs*(gamma + 1)*0.5 - a0) + 0.5*(gamma + 1)*(t - tau)*a_s + vs
    return df


def Exact_Solution(t,  xx, fluid_info):

    N = len(xx)
    uu = np.zeros([N,3])
    L = xx[-1]

    for i in range(int(N)):

        uu[i,:] = Exact_Solution_Point(t,xx[i],L, fluid_info)


    return uu


def Receding_Result(t_end ,xx, fluid_info):
    L = xx[-1]


    exact_uu = Exact_Solution(t_end, xx, fluid_info)

    plt.figure(1)
    plt.plot(xx, exact_uu[:, 0], 'r-', label='exact_desity')

    plt.legend(loc='upper right')
    plt.title('receding forced motion L = %d' % (L))

    plt.figure(2)
    plt.plot(xx, exact_uu[:, 1], 'r-', label='exact_v')
    plt.legend(loc='upper right')
    plt.title('receding forced motion L = %d' % (L))

    plt.figure(3)

    plt.plot(xx, exact_uu[:, 2], 'r-', label='exact_p')
    plt.legend(loc='upper right')
    plt.title('receding forced motion L = %d' % (L))
    plt.show()
