# 1D-Porous-FIVER


This is 1D fluid test problem for invisid flow
inflow fluid@left ** porous membrane ** fluid@right outflow
I will compare different FIVER schemes


1. original FIVER, original porous treatment
2. Dante's FIVER with ghost interpolation, original porous treatment
3. To discuss, FIVER with Darcy's law



Time discretization


Problem description:
	This is flexible piston problem, with closed end at left and flexible piston at right. Tube length is L, and xs_0 = L/2, and the equation of motion is simply that of a spring:
	ms \dot\dot{xs} + ks(xs - xs_0) = f(t) = p_B

The mesh is vertex centered grid,
        e0     e1     e2     e3     e4.....
    v0      v1     v2     v3     v4 .....
v is vertex, e is dual cell boundary(edge)

First try staggered  A6, explicit fluid and implicit structure method.
                      s_{n + 0.5}
           s^p_{n}                   s^p_{n+1}
           f_n                       f_{n+1}



class fluid has two element:
    xx: the grid information
    ww_n: fluid state variables [rho, rho u, rho e]
    ww_old_n: fluid state variables of last step ww_{n - 1}

class structure has two element:
    qq_n: structure state variables [u, dot u, dot dot u]
    qq_old_n: q_{n - 1}
    which are both predicted value at these time points




   For structure subsystem, using second order middle point rule update from s_{n+0.5} to s_{n+1.5}
       p^{n+1} = Riemann(w^{n+1}, xs^{n+1}), which need to be reconsidered in porous material or viscous simulation



   Send s^{p}_{n+1} to fluid


   For fluid subsystem, using second order RK, with GTR, RTG in the second substep
       k^1 = F(w^n, x^n, \dot{x}^n)
       w^{n+0.5} = w^n - dt*k^1 , phase change update based on s^{p}_{n+1} to fluid
       k^2 = F(w^n - dtk^1, x^{n+1}, \dot{x}^{n+1})

       w^{n+1} = w^n - dt/2*(k^1 + k^2)
       Real phase change update for fluid

   Send p^{n+1}(s^{p}_{n+1}) to structure subsystems