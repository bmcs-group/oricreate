'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    HasStrictTraits, Dict, \
    Property, Str, Color, Float, \
    on_trait_change, Int, Tuple

import mayavi.mlab as \
    m2_lab
import numpy as np
from viz3d import \
    Viz3D


class FormingTaskView3D(HasStrictTraits):
    '''Target object of a visualization of a forming task.
    '''
    viz3d_dict = Dict(Str, Tuple(Viz3D, Int))
    '''Dictionary of visualization objects.
    '''

    vis3d_list = Property
    '''Gather all the visualization objects
    '''

    def _get_vis3d_list(self):
        return np.unique(np.array([viz3d.vis3d for viz3d in self.viz3d_list]))

    vot = Float(0.0)
    '''Visual object time - specifying the current appearance state of an object
    in the television.
    '''
    @on_trait_change('vot')
    def _broadcast_vot(self):
        for vis4d in self.vis3d_list:
            vis4d.vot = self.vot

    def get_center(self):
        '''Get the center of the displayed figure.
        '''
        x_min, x_max = self._get_extent()
        if x_min != None and x_max != None:
            x_c = (x_min + x_max) / 2.0
        else:
            return [0, 0, 0]
        return x_c

    def _get_extent(self):
        '''Get the lower left front corner and the upper right back corner.
        '''
        mm = []
        if len(self.viz3d_list) == 0:
            raise IndexError, 'vizualization module inserted into fold television'
        for viz3d in self.viz3d_list:
            x_min, x_max = viz3d.min_max
            if x_min != None and x_max != None:
                mm.append([x_min, x_max])
        bnodes = np.array(mm, dtype='float_')
        bb_min, bb_max = np.min(bnodes[:, 0, :], axis=0), np.max(
            bnodes[:, 1, :], axis=0)
        return bb_min, bb_max

    xyz_grid_resolution = Int(10)
    '''Resolution of the field grid within the extent for 
    visualization of scalar fields.
    '''

    xyz_grid = Property
    '''Get the coordinates for a point grid within the visualized extent.
    used for visualization of levelset functions - target surface.
    '''

    def _get_xyz_grid(self):
        X0, X1 = np.array(self._get_extent(), dtype='d')
        extension_factor = 2
        d = np.fabs(X1 - X0) * extension_factor
        x0 = X0 - 0.1 * d - 0.1
        x1 = X1 + 0.1 * d + 0.1
        ff_r = complex(0, self.xyz_grid_resolution)
        x, y, z = np.mgrid[x0[0]:x1[0]:ff_r,
                           x0[1]:x1[1]:ff_r,
                           x0[2]:x1[2]:ff_r]
        return x, y, z * 2.0

    bgcolor = Color((1.0, 1.0, 1.0))
    fgcolor = Color((0.0, 0.0, 0.0))

    mlab = Property(depends_on='input_change')
    '''Get the mlab handle'''

    def _get_mlab(self):
        return m2_lab

    def add(self, viz3d, order=1):
        '''Add a new visualization objectk.'''
        viz3d.ftv = self
        label = '%s(%s)' % (viz3d.label, str(viz3d.__class__))
        if self.viz3d_dict.has_key(label):
            raise KeyError, 'viz3d object named %s already registered' % viz3d.label
        self.viz3d_dict[label] = (viz3d, order)

    viz3d_list = Property

    def _get_viz3d_list(self):
        map_order_viz3d = {}
        for idx, (viz3d, order) in enumerate(self.viz3d_dict.values()):
            map_order_viz3d['%5g%5g' % (order, idx)] = viz3d
        return [map_order_viz3d[key] for key in sorted(map_order_viz3d.keys())]

    def plot(self):
        '''Plot the current visualization objects.
        '''
        fig = self.mlab.gcf()
        self.mlab.figure(fig, bgcolor=self.bgcolor, fgcolor=self.fgcolor)
        for viz3d in self.viz3d_list:
            viz3d.plot()

    def update(self, vot=0.0, force=False):
        '''Update current visualization.
        '''
        self.vot = vot
        print 'FTV vot', vot
        fig = self.mlab.gcf()
        fig.scene.disable_render = True
        for viz3d in self.viz3d_list:
            if force:
                viz3d.vis3d_changed = True
            viz3d.update()
        fig.scene.disable_render = False

    def show(self, *args, **kw):
        '''Render the visualization.
        '''
        self.mlab.show(*args, **kw)


FTV = FormingTaskView3D

if __name__ == '__main__':

    from visual3d import Visual3D

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
