__author__ = 'zhengyuh'


import numpy as np
import copy as copy
import matplotlib.pyplot as plt
from Structure import Structure
from Fluid import Fluid
class Intersector_Base:
    def __init__(self, fluid, structure):

        self.qq_n = structure.qq_n

        self.nverts = fluid.nverts

        self.nedges = fluid.nedges

        self.L = fluid.L

        self.dx = self.L/self.nedges

        self.verts = fluid.verts

        self.status = np.empty(self.nverts, dtype=bool)

        self.intersect_or_not = np.empty(self.nedges, dtype=bool)

        self._update(structure.qq_n)

    def _update(self, qq_n):

        self.qq_n_old = self.qq_n

        self.qq_n = qq_n

        x = qq_n[0] #current structure position

        dx = self.dx

        self.intersect_id = intersect_id = int(x/dx)

        self.intersect_id_old = self.intersect_id

        self.intersect_or_not[:] = False

        self.intersect_or_not[self.intersect_id] = True

        self.status[:] = False

        self.status[0:self.intersect_id+1] = True





    def _phase_change(self, W0):
        intersect_id_old = self.intersect_id_old
        intersect_id = self.intersect_id
        if(intersect_id > intersect_id_old + 1):
            print('ERROR: move too fast!!!')
        if(intersect_id > intersect_id_old):
            W0[intersect_id] = 2*W0[intersect_id - 1] - W0[intersect_id - 2]


    def _velocity(self):
        return self.qq_n[1]

    def _position(self):
        return self.intersect_id, self.qq_n[0]

