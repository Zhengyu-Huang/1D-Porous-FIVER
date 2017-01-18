import numpy as np
import matplotlib.pyplot as plt
from Limiter import Limiter
from Fluid import Fluid
from Structure import Structure
from Intersector import Intersector
import Utility
import Flux

class Embedded_Explicit_Solver_Base:
    def __init__(self, fluid, structure, time_info):
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

        self.intersector = Intersector(fluid,structure)

        self.limiter = Limiter('Van_Albada')

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

        dt = self.dt

        print('start A6 method')

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

        intersector._phase_change(W0)

        #self._check_solution(W0);

        k2[:, :] = 0

        Utility.conser_to_pri_all(W0, V,gamma)

        #print('2nd density', V[:, 0])

        #print('2nd pressure', V[:, 2])


        self._compute_RK_update(V, k2);

        R = 1.0 / 2.0 * (k1 + k2)

        W += R * dt / control_volume;


        intersector._phase_change(W) #use only intersector information

        #self._check_solution(W);

        #self._compute_residual(R, W)




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

        porous = self.structure.porous_ratio

        gamma = fluid.gamma

        self._lsq_gradient(V)

        for i in range(fluid.nedges):
            # x_{i-1}              x_i                   x_{i+1}
            #           e_{i-1}            e_i

            v_l, v_r = V[i, :], V[i+1, :]

            l_active, r_active = status[i], status[i+1]

            intersect = intersect_or_not[i]

            e_lr = fluid.edge_vector[i]


            dv_l, dv_r = np.dot(e_lr, self.gradient_V[i, :]), -np.dot(e_lr, self.gradient_V[i+1, :])

            if (l_active and r_active and not intersect):

                v_ll, v_rr = limiter._reconstruct(v_l, v_r, dv_l, dv_r)

                #if(v_ll[2] < 0 or v_rr[2] < 0 or v_ll[0] < 0  or v_rr[0] <0):
                #    print('stop debug')
                #    limiter._reconstruct(v_l, v_r, dv_l, dv_r)
                flux = Flux._Roe_flux(v_ll, v_rr, gamma)



                R[i, :] -= flux

                R[i+1, :] += flux

            else:
                #TODO
                #############################################################
                # consider there are two case,
                # consider that the membrane is porous
                # 1. ghost update
                # 2. compute flux regarding there is no structure
                # 3. compute fluid structure flux
                # 4. average these two fluxes
                #############################################################

                v_ll, v_rr = v_l,v_r
                #print('FIVER', v_l, v_r)
                flux = Flux._Roe_flux(v_ll, v_rr, gamma)



                if (l_active):


                    v_Rr = self.FIVER(V, i , i, limiter)

                    flux_FS = Flux._Roe_flux(v_ll, v_Rr, gamma)

                    R[i, :] -= porous*flux + (1-porous)*flux_FS

                if (r_active):

                    v_Rl = self.FIVER(V, i , i+1, limiter)

                    flux_FS = Flux._Roe_flux(v_Rl, v_rr, gamma)

                    R[i+1, :] += porous*flux + (1-porous)*flux_FS


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
                W_oo = fluid.W_oo


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





    def _draw(self):
        fluid = self.fluid
        V = np.empty(shape=[fluid.nverts, 3], dtype=float)

        W = self.W

        L = fluid.L

        gamma = fluid.gamma

        verts = fluid.verts

        Utility.conser_to_pri_all(W, V, gamma)

        xs = self.structure.qq_n[0]

        plt.figure(1)
        plt.plot(verts, V[:,0], 'r-', label = 'desity')
        plt.plot((xs, xs), (0, 1), 'k-')
        plt.legend(loc='upper right')
        plt.title('receding forced motion L = %d' % (L))

        plt.figure(2)
        plt.plot(verts, V[:, 1], 'r-', label = 'v')
        plt.plot((xs, xs), (0, 1), 'k-')
        plt.legend(loc='upper right')
        plt.title('receding forced motion L = %d' % (L))

        plt.figure(3)

        plt.plot(verts, V[:, 2], 'r-',label = 'p')
        plt.plot((xs, xs), (0, 1), 'k-')
        plt.legend(loc='upper right')
        plt.title('receding forced motion L = %d' % (L))
        #plt.show()




