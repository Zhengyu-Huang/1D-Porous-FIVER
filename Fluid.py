import numpy as np
import Utility

class Fluid:
    def __init__(self, L,N,fluid_info, boundary_condition):
        '''
        This is fluid class saves only fluid mesh information

        :param L: fluid domain size, [0,L]
        :param N: number of verts
        '''

        self.nverts = N

        self.nedges = N-1

        self.L = L

        self.verts = verts = np.linspace(0, L, N)

        self.control_volume = np.zeros(shape = [N,1],dtype = float)
        for i in range(N-1):
            self.control_volume[i] += (verts[i + 1] - verts[i])/2.0

            self.control_volume[i + 1] += (verts[i + 1] - verts[i])/2.0

        self. edge_vector = np.zeros(shape = [self.nedges],dtype = float)

        for i in range(N-1):

            self.edge_vector[i] = verts[i + 1] - verts[i]

        self.W = np.empty(shape=[N, 3], dtype=float)

        self.bc = boundary_condition

        self._init_states(fluid_info)
        #initialize fluid conservative variables

    def _init_states(self, fluid_info):
        N = self.nverts
        L = self.L
        if(fluid_info[0] == 'constant'):

            gamma,rho,v,p = fluid_info[1:]

            self.gamma = gamma

            w = Utility.pri_to_conser([rho,v,p],gamma)

            for i in range(N):

                self.W[i,:] = w

            self.Wl_oo = self.Wr_oo = w



        elif(fluid_info[0] == 'shock_tube'):

            gamma, rho_l, v_l, p_l, rho_r, v_r, p_r = fluid_info[1:]

            self.gamma = gamma

            w_l = Utility.pri_to_conser([rho_l, v_l, p_l], gamma)

            w_r = Utility.pri_to_conser([rho_r, v_r, p_r], gamma)

            for i in range(N):

                self.W[i,:] = w_l if self.verts[i] <= L/2.0 else w_r #todo define the left and right

            self.Wl_oo,self.Wr_oo = w_l,w_r

        elif(fluid_info[0] == 'smooth'):
            # This is smooth initialization, as Alex Main's Thesis
            verts = self.verts
            
            gamma, rho, v = 1.4, 1.0, 0.0

            self.gamma = gamma

            # level set initial position
            x_0 = L / 2

            for i in range(N):

                if (verts[i] <= x_0):

                    if (verts[i] >= L / 2.0 - 0.2):
                        p = 1.0e6 * (L / 2.0 - 0.2 - verts[i]) ** 4 * (L / 2.0 + 0.2 - verts[i]) ** 4 + 1.0
                    else:
                        p = 1.0

                else:
                    if (verts[i] <= L / 2.0 + 0.2):

                        p = 1.0e6 * (L / 2.0 - 0.2 - verts[i]) ** 4 * (L / 2.0 + 0.2 - verts[i]) ** 4 + 1.0
                    else:
                        p = 1.0

                w = Utility.pri_to_conser([rho, v, p], gamma)

                self.W[i, :] = w


        self.mass = np.dot(self.W[:,0], self.control_volume)
        self.momentum = np.dot(self.W[:,1], self.control_volume)
        self.energy = np.dot(self.W[:,2], self.control_volume)






