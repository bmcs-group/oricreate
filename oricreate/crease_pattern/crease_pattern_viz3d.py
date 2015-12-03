'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    Array, Tuple, Property, Bool, Float

import numpy as np
from oricreate.viz3d import Viz3D


class CreasePatternViz3D(Viz3D):
    '''Visualize the crease Pattern
    '''
    N_selection = Array(int)
    L_selection = Array(int)
    F_selection = Array(int)

    lines = Bool(True)

    N_L_F = Property(Tuple)
    '''Geometry with applied selection arrays.
    '''

    def _get_N_L_F(self):
        vis3d = self.vis3d
        x, L, F = vis3d.x, vis3d.L, vis3d.F
        if len(self.N_selection):
            x = x[self.N_selection]
        if len(self.L_selection):
            L = L[self.L_selection]
        if len(self.F_selection):
            F = F[self.F_selection]
        return x, L, F

    def plot(self):

        m = self.ftv.mlab
        N, L, F = self.N_L_F
        x, y, z = N.T
        if len(F) > 0:
            cp_pipe = m.triangular_mesh(x, y, z, F,
                                        line_width=3,
                                        color=(0.6, 0.6, 0.6))
            if self.lines is True:
                cp_pipe.mlab_source.dataset.lines = L
                tube = m.pipeline.tube(cp_pipe,
                                       tube_radius=self._get_line_width())
                m.pipeline.surface(tube, color=(1.0, 1.0, 1.0))

        else:
            cp_pipe = m.points3d(x, y, z, scale_factor=0.2)
            cp_pipe.mlab_source.dataset.lines = L
        self.cp_pipe = cp_pipe

    def update(self):
        N = self.N_L_F[0]
        self.cp_pipe.mlab_source.set(points=N)

    def _get_bounding_box(self):
        N = self.N_L_F[0]
        return np.min(N, axis=0), np.max(N, axis=0)

    def _get_max_length(self):
        return np.linalg.norm(self._get_bounding_box())

    line_with_factor = Float(0.004)

    def _get_line_width(self):
        return self._get_max_length() * self.line_with_factor
