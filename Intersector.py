#This is intersector class
import numpy as np
import Utility
class Intersector:
    def __init__(self,fluid,structure,tolerance = 1e-10):
        """
        Attributes:
            status:  bool[number of fluid vertices], true if the vertex  is real
            intersect_or_not: bool[number of fluid edges], true if the fluid edge intersects with structure
            intersect_result: (float, int, float, float, int , float)[number of fluid edges],
                              alpha_1 ,s1, beta_1 ,alpha_2 ,s2, beta_2
                              alpha, save the intersection point information for fluid edges, saying fluid edge (x1, x2)
                              alpha1 is for intersecting point from left to right, which is  x1 + alpha1*(x2 - x1), which is also the point (s1, beta1) on structure
                              alpha1 is for intersecting point from right to left, which is  x2 + alpha2*(x1 - x2), which is also the point (s2, beta2) on structure

        """

        self.fluid = fluid
        self.structure = structure

        self.emb_n, self.emb_old_n = np.empty(shape=2,dtype=float),np.empty(shape=2,dtype=float)

        self.emb_n[0],self.emb_n[1]= structure.qq_n
        self.emb_old_n[0], self.emb_old_n[1] = structure.qq_old_n

        self.L = fluid.L
        self.nverts = nverts = self.fluid.nverts
        self.nedges = nedges = self.fluid.nedges

        self.tolerance = tolerance


        #ghost node is flase
        self.status = [True] * nverts
        self.intersect_or_not = [False] * nedges

        xs = self.emb_n[0]

        dx = float(self.L)/nedges

        self.intersect_edge_id = int(np.floor((xs + self.tolerance)/dx))

        self.intersect_or_not[self.intersect_edge_id] = True




    def _update(self,qq_n):


    #update the embedded position to qq_n
    #update phase change nodes


        self.emb_old_n[0], self.emb_old_n[1] = self.emb_n

        self.emb_n[0], self.emb_n[1] = qq_n

        self.intersect_edge_id_old = intersect_edge_id_old = self.intersect_edge_id

        xs = self.emb_n[0]

        dx = float(self.L)/self.nedges

        self.intersect_edge_id = int(np.floor((xs + self.tolerance)/dx))


        if self.intersect_edge_id != intersect_edge_id_old:
            #update ghost edge
            self.intersect_or_not[intersect_edge_id_old] = False
            self.intersect_or_not[self.intersect_edge_id] = True





    def _phase_change(self,W):
        verts = self.fluid.verts
        gamma = self.fluid.gamma
        intersect_edge_id = self.intersect_edge_id
        intersect_edge_id_old = self.intersect_edge_id_old

        if intersect_edge_id != intersect_edge_id_old:
            #phase change node

            if(self.intersect_edge_id == intersect_edge_id_old + 1):
                #populate node self.intersect_edge_id
                i = self.intersect_edge_id
                W[i,:] = Utility.interpolation(W[i-1,:],verts[i-1], W[i-2,:], verts[i-2], verts[i],gamma)

            elif(self.intersect_edge_id == intersect_edge_id_old - 1):
                #populate node intersect_edge_id_old
                i = intersect_edge_id_old
                W[i,:] = Utility.interpolation(W[i+1,:],verts[i+1], W[i+2,:], verts[i+2], verts[i],gamma)

            else:
                print('move too many grids in one step')




    def _position(self):

        return [self.intersect_edge_id, self.emb_n[0]]

    def _velocity(self):
        return self.emb_n[1]


