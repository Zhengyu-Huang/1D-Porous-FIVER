__author__ = 'zhengyuh'
import numpy as np

def _check_solution(diskCase, rho_l, v_l, p_l,rho_r,  v_r,  p_r, v_m,   p_m, dp_l, dp_r, gamma, DEBUG = False):
    #rho_mr, double &rho_ml,bool DEBUG)
    '''
    diskCase 0,1,2,3, i means disk is between i wave and i+1 wave;
    left fluid state variable rho_l, v_l, p_l;
    right fluid state variable rho_r, v_r, p_r;
    velocity and pressure after the 1- wave are v_m, p_m + dp_l
    velocity and pressure before the 3- wave are v_m, p_m + dp_r
    density before and after contact discontinuity are rho_ml, rho_mr
    '''
    if(p_r <= 0.0 or p_l <= 0.0 or rho_l <= 0.0 or rho_r <= 0.0 or p_m + dp_r <= 0.0 or p_m+dp_l <= 0.0):
        return 0., 0., False;

    result = True;
    a_l, a_r = np.sqrt(gamma * p_l / rho_l),np.sqrt(gamma * p_r / rho_r); #sound speed

    #Left side
    p_ml = p_m + dp_l;
    if (p_ml < p_l): #left rarefaction wave
        s_l = v_l - a_l;
        a_ml = a_l * (p_ml / p_l)**((gamma - 1) / (2 * gamma))
        s_ml = v_m - a_ml;
        rho_ml = rho_l * (p_ml/ p_l)**(1 / gamma);
        if(DEBUG): 
            print("DEBUG: case %d, left rarefaction, velocity s_l s_ml are %f and %f\n" %(diskCase, s_l, s_ml))
        if((diskCase == 0 and s_l < 0.0 and s_ml < 0.0)or(diskCase == 1 and s_l > 0.0 and s_ml > 0.0)
           or(diskCase == 2 and s_l > 0.0 and s_ml > 0.0) or(diskCase == 3 and s_l > 0.0 and s_ml > 0.0)):
            result = False;

  
    else: #left shock wave
        s_shock = v_l - a_l * np.sqrt((gamma + 1) *p_ml / (2 * gamma * p_l) + (gamma - 1) / (2 * gamma));
        rho_ml = rho_l * (p_ml/ p_l + (gamma - 1) / (gamma + 1)) / ((gamma - 1) * p_ml/ ((gamma + 1) * p_l) + 1);
        if(DEBUG):
            print("DEBUG: case %d, left shock, velocity s_shock_l is %f\n" %(diskCase, s_shock))
        if((diskCase == 0 and s_shock < 0.0 )or(diskCase == 1 and s_shock > 0.0 )or(diskCase == 2 and s_shock > 0.0)
           or (diskCase == 3 and s_shock > 0.0)):
            result = False;

    if(DEBUG):
        print("DEBUG: case %d, contact discontinuity velocity is %f\n"% (diskCase, v_m))
    if((diskCase == 0 and v_m < 0.0 )or(diskCase == 1 and v_m < 0.0 )or (diskCase == 2 and v_m > 0.0)
       or(diskCase == 3 and v_m > 0.0)):
        result = False;

    #Right side
    p_mr = p_m + dp_r;
    if (p_mr < p_r): #right rarefaction wave

        s_r = v_r + a_r;
        a_mr = a_r * (p_mr / p_r)**((gamma - 1) / (2 * gamma));
        s_mr = v_m + a_mr;
        rho_mr = rho_r * (p_mr / p_r)**(1 / gamma);
        if(DEBUG):
            print("DEBUG: case %d,  right rarefaction, velocity s_mr s_r are %f and %f\n"%(diskCase, s_mr, s_r))
        if((diskCase == 0 and s_r < 0.0 and s_mr < 0.0)or(diskCase == 1 and s_r < 0.0 and s_mr < 0.0)
           or (diskCase == 2 and s_r < 0.0 and s_mr < 0.0) or(diskCase == 3 and s_r > 0.0 and s_mr > 0.0)):
            result = False;


    else:   #right shock wave
        s_shock = v_r + a_r * np.sqrt((gamma + 1) * p_mr / (2 * gamma * p_r) + (gamma - 1) / (2 * gamma));
        rho_mr = rho_r * (p_mr / p_r + (gamma - 1) / (gamma + 1)) / ((gamma - 1) * p_mr / ((gamma + 1) * p_r) + 1);
        if(DEBUG):
            print("DEBUG: case %d, right shock, velocity s_shock_r is %f\n" %(diskCase, s_shock))
        if((diskCase == 0 and s_shock < 0.0 )or(diskCase == 1 and s_shock < 0.0 )
           or (diskCase == 2 and s_shock < 0.0)or (diskCase == 3 and s_shock > 0.0)):
            result = False;


    return rho_ml, rho_mr, result;


def _pressure_function(p, rho_k,  p_k,  a_k,  A_k,  B_k, gamma):


    if(p > p_k):

        f_k = (p - p_k) * np.sqrt(A_k / (p + B_k));

        df_k = np.sqrt(A_k / (B_k + p)) * (1 - (p - p_k) / (2 * (B_k + p)));
    else:


        f_k = 2 * a_k / (gamma - 1) * ( (p / p_k)**((gamma - 1) / (2 * gamma)) - 1);

        df_k = 1 / (rho_k * a_k) * (p / p_k)**( -(gamma + 1) / (2 * gamma));

    return f_k , df_k

def _solve_contact_discontinuity(rho_l, v_l, p_l, rho_r, v_r, p_r, dp_l, dp_r, gamma):
    '''

    he pressure and density at contact discontinuity is p_m, v_m
    depends on the position of the actuator disk we have
    The pressure after  the first wave is p_m + dp_l
    The pressure before the third wave is p_m + dp_r
    p is the pressure at contact discontinuity
    '''

    MAX_ITE = 100;
    TOLERANCE = 1.0e-8
    found = False



    d_v = v_r - v_l
    a_l,a_r = np.sqrt(gamma * p_l / rho_l) ,  np.sqrt(gamma * p_r / rho_r)
    A_l,A_r = 2 / ((gamma + 1) * rho_l),  2 / ((gamma + 1) * rho_r)
    B_l,B_r = (gamma - 1) / (gamma + 1) * p_l, (gamma - 1) / (gamma + 1) * p_r

    p_old = (p_l + p_r)/2.0  - min(0, dp_l, dp_r);

    for i in range(MAX_ITE):

        f_l, df_l = _pressure_function(p_old + dp_l, rho_l, p_l, a_l, A_l, B_l, gamma );
        f_r, df_r = _pressure_function(p_old + dp_r, rho_r, p_r, a_r, A_r, B_r, gamma );

        alpha = 1.0
        while(True):
            p_m_cand = p_old - alpha*(f_l + f_r + d_v) / (df_l + df_r);
            if (p_m_cand  <= 0.0 or p_m_cand + dp_l <= 0.0 or p_m_cand + dp_r <= 0.0):
                alpha /= 2.0
            else:
                p_m = p_m_cand
                break

        if (np.fabs(f_l + f_r + d_v) < TOLERANCE):
            found = True;
            break;

        p_old = p_m;

    if (not found):
        print(" ***ERROR: Divergence in Newton-Raphason iteration in Actuator disk Riemann solver",
              rho_l, v_l, p_l, rho_r, v_r, p_r, dp_l, dp_r,);


    v_m = 0.5 * (v_l + v_r + f_r - f_l);
    return v_m, p_m,  found;





def solve_actuator_disk(u_ll, u_rr, dp, gamma):
    rho_l, v_l, p_l = u_ll
    rho_r, v_r, p_r = u_rr

    is_solution = False
    M_l = v_l/np.sqrt(gamma*p_l/rho_l);
    LEFT, CENTER_LEFT,CENTER_RIGHT,RIGHT = 0,1,2,3;
    #first wave, actuator disk, contact wave,  third wave
    #rho_l ,      rho_a,          rho_a=rho_ml   rho_mr         rho_r
    #v_l,         v_a             v_a  =v_ml     v_mr           v_r
    #p_l          p_a             p_a+dp =p_ml   p_mr           p_r
    v_m,p_m, err1 = _solve_contact_discontinuity(rho_l, v_l, p_l, rho_r, v_r, p_r, -dp, 0, gamma)
    rho_ml, rho_mr, is_solution = _check_solution(CENTER_LEFT, rho_l, v_l, p_l, rho_r, v_r, p_r, v_m, p_m,-dp,0,gamma)
    if(is_solution):
        rho_a = rho_ml;
        v_a = v_m;
        p_a = p_m - dp;
        return rho_a,v_a,p_a,is_solution


    #actuator disk, first wave, contact wave, third wave
    #rho_a=rho_l,   rho_a,      rho_ml   rho_mr         rho_r
    #v_a=v_l        v_a           v_ml     v_mr           v_r
    #p_a=p_l        p_a+dp        p_ml     p_mr           p_r
    v_m,p_m, err1 = _solve_contact_discontinuity(rho_l, v_l, p_l+dp, rho_r,v_r,p_r, 0,0,gamma)
    rhom_ml, rho_mr, is_solution = _check_solution(LEFT, rho_l, v_l, p_l + dp, rho_r, v_r, p_r,v_m, p_m,0 ,0,gamma)
    if(is_solution):
        rho_a = rho_l;
        v_a = v_l;
        p_a = p_l;
        return rho_a,v_a,p_a,is_solution



    #first wave, contact wave, actuator disk, third wave
    #rho_l ,      rho_ml,      rho_mr=rho_a    rho_a         rho_r
    #v_l,         v_ml           v_mr=v_a      v_a=v_ml       v_r
    #p_l          p_ml           p_mr=p_a      p_a + dpl      p_r
    v_m,p_m, err1 = _solve_contact_discontinuity(rho_l, v_l, p_l, rho_r, v_r, p_r, 0, dp, gamma)
    rho_ml, rho_mr, is_solution = _check_solution(CENTER_RIGHT, rho_l, v_l, p_l, rho_r, v_r, p_r,v_m, p_m,0 ,dp,gamma)
    if(is_solution):
        rho_a = rho_mr
        v_a = v_m;
        p_a = p_m;
        return rho_a,v_a,p_a,is_solution


    #first wave, contact wave, third wave ,actuator disk
    #rho_l ,      rho_ml      rho_mr         rho_a=rho_r   rho_r
    #v_l,         v_ml        v_mr           v_a=v_r        v_r
    #p_l          p_ml        p_mr           p_a=p_r-dp     p_r
    if(p_r > dp):
        v_m,p_m, err1 = _solve_contact_discontinuity(rho_l, v_l, p_l, rho_r, v_r, p_r - dp, 0, 0, gamma)
        rho_ml, rho_mr, is_solution = _check_solution(RIGHT, rho_l, v_l, p_l, rho_r, v_r, p_r-dp, v_m, p_m,0 ,dp,gamma)
        if(is_solution):
            rho_a = rho_r;
            v_a = v_r;
            p_a = p_r - dp;
            return rho_a,v_a,p_a,is_solution


    print('***ERROR: Actuator disk Riemann solver has no solution, use approximate solution');
    if( M_l >= 0.0 ):
        rho_a = rho_l;
        v_a = v_l;
        p_a = p_l;
    else:
        rho_a = rho_r;
        v_a = v_r;
        p_a = p_r - dp;




    return rho_a,v_a,p_a,is_solution


if __name__ == "__main__":

    gamma = 1.4;
    dp = 0.2
    #rho_l,  v_l , p_l =1.0, 0.0, 1.;
    #rho_r,  v_r, p_r = 0.125, 0.0 , 0.1;


    #rho_l,  v_l , p_l = 1.0 , -2.0, 0.4;
    #rho_r,  v_r, p_r = 1.0, 2.0 , 0.4;


    #rho_l,  v_l , p_l = 1.0 , 0.0, 1000.0;
    #rho_r,  v_r, p_r = 1.0, 0.0 , 0.01;

    #rho_l,  v_l , p_l = 1.0 , 0.0, 0.01;
    #rho_r,  v_r, p_r = 1.0, 0.0 , 100;

    #rho_l,  v_l , p_l = 1.0,  -0.3,  2.;
    #rho_r,  v_r, p_r  = 1.0,  -0.3, 3.;


    M = 0.8;
    rho_l,  v_l,  p_l =  0.796388, 1.203498, 0.823813
    rho_r,  v_r,  p_r =  0.833442, 1.378080, 1.756812
    dp = 1.116071 ;


    u_ll, u_rr = [rho_l, v_l, p_l],[rho_r, v_r, p_r]

    rho_a,v_a,p_a,is_solution = solve_actuator_disk(u_ll, u_rr, dp, gamma)


    print(rho_a, ' ', v_a, ' ', p_a)