import numpy as np
import matplotlib.pyplot as plt
from Limiter import Limiter
from Fluid import Fluid
from Structure import Structure
from Intersector_Base import Intersector_Base
import Utility
import Flux
import Propeller_Riemann_Solver

class Embedded_Explicit_Solver_Base:
    def __init__(self, fluid, structure, time_info, limiter = 'Van_Albada'):
        """
        Initialize Embedded Explicit Solver
        Args:
            fluid_domain:  Fluid_Domain class
            structure:     Structure class
            io_data:       Input information
        Attributes:
            intersector: save fluid embedded surface intersection information
            ghost_node: save ghost node variables vx, vy, T ; vx , vy, T the first ghost value is on the same side
            with the fluid node(interpolating with the first ghost_node_stencil);
            the second ghost value is on the other side(interpolating with the first ghost_node_stencil);
        """
        self.fluid = fluid

        self.structure = structure

        self.intersector = Intersector_Base(fluid,structure)

        self.material = structure.material

        self.limiter = Limiter(limiter)

        nverts = fluid.nverts

        self.W = fluid.W

        self.k1 = np.empty(shape=[nverts, 3], dtype=float)

        self.k2 = np.empty(shape=[nverts, 3], dtype=float)

        self.gradient_V = np.empty(shape=[nverts,3],dtype=float)

        self.t_end, self.dt = time_info

        self.t = 0



    def _lsq_gradient(self, V):

        verts = self.fluid.verts
        nverts = self.fluid.nverts
        status = self.intersector.status
        intersect_or_not = self.intersector.intersect_or_not

        for i in range(nverts):
            # x_{i-1}              x_i                   x_{i+1}
            #           e_{i-1}            e_i
            if(not status[i]):
                continue
            det = 0
            b = np.zeros(shape=[3], dtype=float)
            if(i-1 >= 0 and status[i-1] and not intersect_or_not[i-1]):
                dx = verts[i - 1] - verts[i]

                det += dx**2

                b += (V[i - 1,:] - V[i,:])*dx

            if(i+1 < nverts and status[i+1] and not intersect_or_not[i]):
                dx = verts[i + 1] - verts[i]

                det += dx ** 2

                b += (V[i + 1, :] - V[i, :])*dx

            assert np.fabs(det) > 1e-16

            self.gradient_V[i, :] = b/det



    def _solve(self):
        '''
        A6 strongly coupling scheme
        s1     s2      s3
           f1      f2       f3
        A6_first_step updates fluid to time dt/2
        :return:
        '''

        dt = self.dt

        self._A6_first_step(dt/2)

        self.t += dt / 2

        while self.t < self.t_end - 1e-10:
            if(self.t + dt > self.t_end):
                dt = self.t_end - self.t


            self._structure_advance(dt)

            self._fluid_advance(dt)

            self.t += dt

    def _A6_first_step(self,dt):
        """
        Advance fluid to t_{1/2}
        :return:
        """
        f = self._compute_pressure_load()

        self.structure._predict_half_step(f,dt)

        self._fluid_advance(dt)



    def _structure_advance(self,dt):

        structure = self.structure

        f = self._compute_pressure_load()

        structure._move(f, dt)



    def _fluid_advance(self, dt):
        '''
        update fluid state by RK2
        :param dt: time step
        :return:
           s1^p    s2^p     s3^p
           f1      f2       f3
        embedded boundary is at s1^p, structure is at s1.5, fluid is at f1

        use s1^p to compute first RK step
             k^1 = F(w^1, s1^p)
             w^{1.5} = w^1 + dt*k^1 ,
        update embedded boundary to s2^p
        phase change update based on s2^p to W^{1.5}
             k^2 = F(w^1.5, s_2^p)
             w^2 = w^1 + dt/2*(k^1 + k^2)
        phase change update on s2^p to W^{2}
        '''

        fluid = self.fluid

        structure = self.structure

        intersector = self.intersector

        W, k1, k2 = self.W, self.k1, self.k2



        gamma = fluid.gamma

        #######################################################
        # update fluid, from t_{n-1/2} to t_{n+1/2}
        #####################################################

        V = np.empty(shape=[fluid.nverts, 3], dtype=float)


        Utility.conser_to_pri_all(W, V, gamma)

        #print('1st density', V[:, 0])

        #print('1st pressure', V[:, 2])

        control_volume = fluid.control_volume

        k1[:, :] = 0

        self._compute_RK_update(V, k1);

        W0 = W + k1 * dt / control_volume;

        #update embedded surface to time n+1
        intersector._update(structure.qq_n)
        #print(self.intersector.status[49:51])

        intersector._phase_change(W0)

        #self._check_solution(W0);

        k2[:, :] = 0

        Utility.conser_to_pri_all(W0, V,gamma)

        #print('2nd density', V[:, 0])

        #print('2nd pressure', V[:, 2])


        self._compute_RK_update(V, k2);

        R = 1.0 / 2.0 * (k1 + k2)

        W += R * dt / control_volume;


        intersector._phase_change(W)


        self._conservation_error(W,self.t)




        # self._draw_residual(R)

    def _compute_RK_update(self, V, R):

        self._euler_flux_rhs(V, R)

        self._euler_boundary_flux(V, R)



    def _euler_flux_rhs(self, V, R):
        # convert conservative state variable W to primitive state variable V

        fluid = self.fluid

        status = self.intersector.status

        intersect_or_not = self.intersector.intersect_or_not

        limiter = self.limiter

        gamma = fluid.gamma

        self._lsq_gradient(V)


        for i in range(fluid.nedges):
            # x_{i-1}              x_i                   x_{i+1}
            #           e_{i-1}            e_i

            u_l, u_r = V[i, :], V[i+1, :]

            l_active, r_active = status[i], status[i+1]

            intersect = intersect_or_not[i]

            e_lr = fluid.edge_vector[i]

            du_l, du_r = np.dot(e_lr, self.gradient_V[i, :]), -np.dot(e_lr, self.gradient_V[i+1, :])

            if (l_active and r_active and not intersect):

                u_ll, u_rr = limiter._reconstruct(u_l, u_r, du_l, du_r)

                #if(v_ll[2] < 0 or v_rr[2] < 0 or v_ll[0] < 0  or v_rr[0] <0):
                #    print('stop debug')
                #    limiter._reconstruct(v_l, v_r, dv_l, dv_r)
                flux = Flux._Roe_flux(u_ll, u_rr, gamma)

                R[i, :] -= flux

                R[i+1, :] += flux

            else:
                if(self.material[0] == 'porous'):
                    #############################################################
                    # consider there are two case,
                    # consider that the membrane is porous
                    # 1. ghost update
                    # 2. compute flux regarding there is no structure
                    # 3. compute fluid structure flux
                    # 4. average these two fluxes
                    #############################################################
                    u_ll, u_rr = u_l,u_r

                    flux = Flux._Roe_flux(u_ll, u_rr, gamma)

                    porous = self.material[1]

                    if (l_active):

                        u_Rr = self.FIVER(V, i , i, limiter)

                        flux_FS = Flux._Roe_flux(u_ll, u_Rr, gamma)

                        R[i, :] -= porous*flux + (1-porous)*flux_FS
                        print('l ',u_ll,u_Rr, flux, flux_FS, porous*flux + (1-porous)*flux_FS)

                    if (r_active):

                        u_Rl = self.FIVER(V, i , i+1, limiter)

                        flux_FS = Flux._Roe_flux(u_Rl, u_rr, gamma)

                        R[i+1, :] += porous*flux + (1-porous)*flux_FS
                        print('r ', u_Rl, u_rr, flux, flux_FS,porous*flux + (1-porous)*flux_FS)

                elif(self.material[0] == 'propeller_new_source'):

                    dp = self.material[1]

                    u_ll, u_rr = u_l, u_r

                    flux = Flux._Roe_flux(u_ll, u_rr, gamma)

                    v_m = (u_l[1] + u_r[1])/2.0


                    vs = self.intersector._velocity()

                    source = np.array([0, dp, gamma/(gamma-1)*dp*v_m])#+ dp*(v_m - vs)/(gamma-1)])
                    #source = np.array([0, dp, dp * v_m])
                    R[i,:] -= flux

                    R[i+1] += flux + source

                elif(self.material[0] == 'propeller_old_source'):

                    dp = self.material[1]

                    u_ll, u_rr = u_l, u_r

                    flux = Flux._Roe_flux(u_ll, u_rr, gamma)

                    v_m = (u_l[1] + u_r[1])/2.0


                    vs = self.intersector._velocity()

                    source = np.array([0, dp, dp*v_m])#+ dp*(v_m - vs)/(gamma-1)])
                    #source = np.array([0, dp, dp * v_m])
                    R[i,:] -= flux

                    R[i+1] += flux + source

                elif (self.material[0] == 'propeller_Riemann'):

                    dp = self.material[1]

                    u_ll, u_rr = u_l, u_r

                    dp = self.material[1]

                    #reture the density velocity and pressure upwing of the actuator disk
                    rho_a, v_a, p_a,_ = Propeller_Riemann_Solver.solve_actuator_disk(u_ll, u_rr, dp, gamma)

                    print(rho_a,v_a,p_a)

                    u_Rl, u_Rr = np.array([rho_a,v_a,p_a]), np.array([rho_a,v_a,p_a + dp])

                    flux_l = Flux._Roe_flux(u_ll, u_Rl, gamma)

                    flux_r = Flux._Roe_flux(u_Rr, u_rr, gamma)

                    #flux_l = Flux._flux(u_Rl, gamma)

                    #flux_r = Flux._flux(u_Rr, gamma)



                    R[i, :] -= flux_l

                    R[i + 1] += flux_r

                else:
                    print('unrecognized material', self.material[0])






    def _euler_boundary_flux(self,V,R):

        fluid = self.fluid
        gamma = fluid.gamma

        # there are only 2 boundary vertices
        for i in range(2):

            type = fluid.bc[i]

            n  = 0    if i == 0  else fluid.nverts - 1 # vertex id

            dr = -1.0 if i == 0  else 1.0 # vertex id

            x = fluid.verts[n]

            prim = V[n,:]

            if(type == 'wall'): #slip_wall
                #weakly impose slip_wall boundary condition

                R[n,:] -= np.array([0.0, prim[2]*dr, 0.0])


            elif(type == 'far_field'): #subsonic_outflow
                # parameterise the outflow state by a freestream pressure rho_f

                W_oo = fluid.Wl_oo if n==0 else fluid.Wr_oo


                R[n,:] -= Flux._Steger_Warming(prim, W_oo, dr, gamma)







    def _compute_pressure_load(self):
        '''
        :param W:
        :return: pressure load at current fluid configuration
        '''
        i,xs = self.intersector._position()
        gamma = self.fluid.gamma
        W = self.W
        verts = self.fluid.verts

        status = self.intersector.status
        #expolation pressure force from left
        if(status[i]):

            p_l = Utility.pressure_load_interpolation(W[i,:],verts[i],W[i-1,:],verts[i-1],xs, gamma)

        else:
            p_l = Utility.pressure_load_interpolation(W[i-1, :], verts[i-1], W[i - 2, :], verts[i - 2], xs, gamma)

        # expolation pressure force from right
        if(status[i+1]):
            p_r = Utility.pressure_load_interpolation(W[i+1, :], verts[i+1], W[i + 2, :], verts[i + 2], xs, gamma)
        else:
            p_r = Utility.pressure_load_interpolation(W[i + 2, :], verts[i + 2], W[i + 3, :], verts[i + 3], xs, gamma)

        return p_l - p_r


    def FIVER(self, V, edge_i, vert_n, limiter = None):
        '''
        Impose transmission condition
        :param V: primitive state variables
        :param i: edge number, edge i has one ghost node and one active node,
        or intersected by embedded surface
        :param limiter: flow limiter
        :param n: compute FIVER flux for node n, do everything on the node n side
        :return:
        Return the primitive state variables on ghost side of the edge.
        '''

        fluid_normal = 1 if vert_n == edge_i else -1

        gamma = self.fluid.gamma

        vs = self.intersector._velocity()

        v_R = Flux._FSRiemann(V[vert_n,:], vs , fluid_normal,gamma)

        return v_R;

    def _extrapolate_inactive_node(self,W):
        return

    def _conservation_error(self, W, t = 0.0):
        control_volume = self.fluid.control_volume
        nu_mass = np.dot(W[:, 0],control_volume)
        nu_momentum = np.dot(W[:, 1], control_volume)
        nu_energy = np.dot(W[:, 2], control_volume)
        print('At %f, mass rel. error is %.15f,momentum rel. err is %f, energy rel. err is %.15f' %(t, (nu_mass - self.fluid.mass)/self.fluid.mass,
                                                                                  (nu_momentum - self.fluid.momentum), #/self.fluid.momentum,
                                                                                  (nu_energy - self.fluid.energy) / self.fluid.energy))



    def _draw(self):

        fluid = self.fluid
        V = np.empty(shape=[fluid.nverts, 3], dtype=float)

        W = self.W

        self._extrapolate_inactive_node(W)

        L = fluid.L

        gamma = fluid.gamma

        verts = fluid.verts

        Utility.conser_to_pri_all(W, V, gamma)



        xs = self.structure.qq_n[0]

        title = self.material[0]

        plt.figure(1)
        plt.plot(verts, V[:,0], 'ro-', markersize=2.0, label = 'rho')
        plt.plot((xs, xs), (0, 1), 'k-')
        plt.legend(loc='upper right')
        plt.title(title)

        plt.figure(2)
        plt.plot(verts, V[:, 1], 'ro-',markersize=2.0, label = 'v')
        plt.plot((xs, xs), (0, 1), 'k-')
        plt.legend(loc='upper right')
        plt.title(title)

        plt.figure(3)

        plt.plot(verts, V[:, 2], 'ro-',markersize=2.0,label = 'p')
        plt.plot((xs, xs), (0, 1), 'k-')
        plt.legend(loc='upper right')
        plt.title(title)

        plt.figure(4)

        plt.plot(verts, V[:, 0]*V[:,1], 'ro-',markersize=2.0,label = 'mass_f')
        plt.plot((xs, xs), (0, 1), 'k-')
        plt.plot(verts, W[:,1]*V[:,1] + V[:,2], 'bo-',markersize=2.0,label = 'momentum_f')
        plt.plot((xs, xs), (0, 1), 'k-')
        plt.plot(verts, (W[:,2]+V[:,2])*(V[:,1]), 'yo-',markersize=2.0,label = 'energy_f')
        plt.plot((xs, xs), (0, 1), 'k-')
        plt.legend(loc='upper right')
        plt.title(title)

        M = self.fluid.nverts//2
        print('dp is ',self.material[1])
        print('pressure difference is %.15f, mass flux is %.15f , %.15f, momentum flux is %.15f , %.15f'
              %(V[M+10,2] - V[M-10,2], V[M-10, 0]*V[M-10,1], V[M+10, 0]*V[M+10,1],W[M-10,1]*V[M-10,1] + V[M-10,2],W[M+10,1]*V[M+10,1] + V[M+10,2] ))

        print('pressure difference is %.15f, stagnation pressure diff is %.15f, mass flux diff is %.15f , momentum flux diff is %.15f'
              %(V[M+10,2] - V[M-10,2], V[M+10,2]  + 0.5*V[M+10,0]*V[M+10,1]**2 - V[M-10,2] - 0.5*W[M-10,0]*V[M-10,1]**2, V[M-10, 0]*V[M-10,1] - V[M+10, 0]*V[M+10,1], W[M-10,1]*V[M-10,1] + V[M-10,2] - W[M+10,1]*V[M+10,1] - V[M+10,2] ))
        print('left rho,u,p,E  is %.15f, %.15f,  %.15f , %.15f' %(V[M-2,0] , V[M-2, 1], V[M-2,2] , W[M-2,2]))
        print('right rho,u,p,E  is %.15f, %.15f,  %.15f, %.15f' %(V[M+2,0] , V[M+2, 1], V[M+2,2] , W[M-2,2]))
        plt.show()


    def _save(self):

        fluid = self.fluid

        V = np.empty(shape=[fluid.nverts, 3], dtype=float)

        W = self.W

        self._extrapolate_inactive_node(W)

        L = fluid.L

        gamma = fluid.gamma

        verts = fluid.verts

        Utility.conser_to_pri_all(W, V, gamma)



        np.save(self.material[0],V)



