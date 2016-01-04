'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasStrictTraits, Dict, \
    Property, Str, Color, Float, \
    on_trait_change

import mayavi.mlab as \
    m2_lab
import numpy as np
from viz3d import \
    Viz3D


class FormingTaskView3D(HasStrictTraits):
    '''Target object of a visualization of a forming task.
    '''
    viz3d_dict = Dict(Str, Viz3D)
    '''Dictionary of visualization objects.
    '''

    vis3d_list = Property
    '''Gather all the visualization objects
    '''

    def _get_vis3d_list(self):
        return np.unique(np.array([viz3d.vis3d for viz3d in self.viz3d_dict.values()]))

    vot = Float(0.0)

    @on_trait_change('vot')
    def _broadcast_vot(self):
        for vis4d in self.vis3d_list:
            vis4d.vot = self.vot

    def get_center(self):
        for vis3d in self.vis3d_list:
            x_min, x_max = np.min(vis3d.x, axis=0), np.max(vis3d.x, axis=0)
            x_c = (x_min + x_max) / 2.0
        return x_c

    bgcolor = Color((1.0, 1.0, 1.0))
    fgcolor = Color((0.0, 0.0, 0.0))

    mlab = Property(depends_on='input_change')
    '''Get the mlab handle'''

    def _get_mlab(self):
        return m2_lab

    def add(self, viz3d):
        '''Add a new visualization objectk.'''
        viz3d.ftv = self
        self.viz3d_dict[viz3d.label] = viz3d

    def plot(self):
        '''Plot the current visualization objects.
        '''
        fig = self.mlab.gcf()
        self.mlab.figure(fig, bgcolor=self.bgcolor, fgcolor=self.fgcolor)
        for viz3d in self.viz3d_dict.values():
            viz3d.plot()

    def update(self, vot=0.0, force=False):
        '''Update current visualization.
        '''
        self.vot = vot
        for viz3d in self.viz3d_dict.values():
            if force:
                viz3d.vis3d_changed = True
            viz3d.update()

    def show(self, *args, **kw):
        '''Render the visualization.
        '''
        self.mlab.show(*args, **kw)


FTV = FormingTaskView3D

if __name__ == '__main__':

    from visual3d import Visual3D
    from traits.api import Tuple
    import numpy as np

    class PointCloud(Visual3D):
        '''State object
        '''
        p = Tuple
        '''Point positions
        '''

        def _p_default(self):
            x, y, z, s = np.random.random((4, 100))
            return x, y, z, s

        def scale(self, c):
            x, y, z, s = self.p
            self.p = x * c, y * c, z * c, s * c

        def _viz3d_dict_default(self):
            return dict(default=PointCloudViz3D(vis3d=self),
                        something_else=PointCloudViz3D(vis3d=self))

    class PointCloudViz3D(Viz3D):
        '''Visualization object
        '''

        def plot(self):
            x, y, z, s = self.vis3d.p
            self._pipe = self.ftv.mlab.points3d(x, y, z, s)

        def update(self):
            x, y, z, s = self.vis3d.p
            points = np.c_[x, y, z]
            self._pipe.mlab_source.set(points=points)

    ftv = FTV()
    pc = PointCloud()
    ftv.add(pc.viz3d)
    ftv.plot()
    ftv.mlab.savefig('xxx01.jpg')
    pc.scale(1.3)
    ftv.update()
    ftv.mlab.savefig('xxx02.jpg')
    pc.scale(1.3)
    ftv.update()
    ftv.mlab.savefig('xxx03.jpg')
    ftv.show()
