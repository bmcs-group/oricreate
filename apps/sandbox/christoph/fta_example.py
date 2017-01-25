'''
Created on 23.01.2017
'''
import numpy as np
from traits.api import Tuple, Property, cached_property
from oricreate.api import FTA, FTV, Viz3D
from oricreate.viz3d.visual3d import Visual3D


class PointCloudViz3D(Viz3D):
        '''Visualization object
        '''

        def plot(self):
            x, y, z, s = self.vis3d.points
            self.pipes['points'] = self.ftv.mlab.points3d(x, y, z, s)

        def update(self):
            x, y, z, s = self.vis3d.points
            points = np.c_[x, y, z]
            pipe = self.pipes['points']
            pipe.mlab_source.set(points=points, scalars=s)

class PointCloud(Visual3D):
    '''State object
    '''
    p = Tuple
    '''Point positions
    '''

    def _p_default(self):
        x = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        y = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        z = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        s = [1, 2, 3, 4, 5, 6, 7, 8]
        return x, y, z, s

    x = Property

    def _get_x(self):
        x, y, z, s = self.p
        return np.c_[x, y, z]

    points = Property(depends_on='+time_change')

    @cached_property
    def _get_points(self):
        x, y, z, s = self.p
        s = np.array(s, float)
        s[-1] *= (1.0 - 0.9 * self.vot)
        print x, y, z, s
        return x, y, z, s

    viz3d_classes = dict(default=PointCloudViz3D,
                         something_else=PointCloudViz3D)


if __name__ == '__main__':

    pc = PointCloud(anim_t_start=0, anim_t_end=40)
    ftv = FTV()
    ftv.add(pc.viz3d['default'])
    ftv.add(pc.viz3d['something_else'])

    fta = FTA(ftv=ftv)
    fta.init_view(a=0, e=0, d=8, f=(0, 0, 0), r=0)
    fta.add_cam_move(e=50, n=20,
                     duration=10,
                     azimuth_move='linear',
                     elevation_move='damped',
                     distance_move='damped')
    fta.add_cam_move(e=0, n=20,
                     duration=30,
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')
    fta.plot()
    fta.configure_traits()
