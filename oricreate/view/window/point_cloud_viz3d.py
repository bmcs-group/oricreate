'''
Created on Feb 16, 2018

@author: rch
'''

import numpy as np
from oricreate.viz3d import \
    Viz3D, Visual3D as Vis3D
import traits.api as tr


class PointCloudViz3D(Viz3D):
    '''Visualization object
    '''

    def plot(self):
        x, y, z, s = self.vis3d.p
        self.pipes['points'] = self.ftv.mlab.points3d(x, y, z, s)

    def update(self):
        x, y, z, s = self.vis3d.points
        points = np.c_[x, y, z]
        self.pipes['points'].mlab_source.set(points=points)


class PointCloud(Vis3D):
    '''State object
    '''
    p = tr.Tuple
    '''Point positions
    '''

    def _p_default(self):
        x, y, z, s = np.random.random((4, 100))
        return x, y, z, s

    points = tr.Property(tr.Tuple)

    def _get_points(self):
        x, y, z, s = self.p
        c = self.vot
        return x * c, y * c, z * c, s * c

    viz3d_classes = dict(default=PointCloudViz3D,
                         something_else=PointCloudViz3D)
