'''
Created on Dec 3, 2015

@author: rch
'''

import numpy as np
from oricreate.viz3d import Viz3D


class FuTargetPsiValueViz3D(Viz3D):
    '''Visualize the crease Pattern
    '''

    def plot(self):

        m = self.ftv.mlab
        cp = self.vis3d.formed_object
        x_t = cp.x

    def update(self):

        cp = self.vis3d.formed_object
        x_t = cp.x

    def _get_bounding_box(self):
        cp = self.vis3d.formed_object
        x_t = cp.x
        return np.min(x_t, axis=0), np.max(x_t, axis=0)

    def _get_max_length(self):
        return np.linalg.norm(self._get_bounding_box())
