'''
Created on Dec 3, 2015

@author: rch
'''

from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from traits.api import \
    HasStrictTraits, Dict, \
    Property, Str, Float, \
    on_trait_change, Int, Tuple, Instance, Bool
from traitsui.api import \
    View, HSplit, VSplit, VGroup, Item, UItem, TableEditor, ObjectColumn, \
    RangeEditor, Tabbed
from tvtk.pyface.scene_editor import SceneEditor

import numpy as np
from oricreate.viz3d.viz3d import \
    Viz3D


def oricreate_mlab_label(m):
    atext_width = 0.15
    m.text(1 - atext_width, 0.003, 'oricreate',
           color=(.7, .7, .7),
           width=atext_width)


viz3d_dict_editor = TableEditor(
    columns=[ObjectColumn(label='Label', name='label', editable=False),
             ObjectColumn(label='Hidden', name='hidden', editable=True),
             ObjectColumn(label='Start', name='anim_t_start', editable=False),
             ObjectColumn(label='End', name='anim_t_end', editable=False),
             ],
    selection_mode='row',
    selected='object.selected_viz3d'
)


class FormingTaskView3D(HasStrictTraits):
    '''Target object of a visualization of a forming task.
    '''
    viz3d_dict = Dict(Str, Instance(Viz3D))
    '''Dictionary of visualization objects.
    '''

    vis3d_list = Property
    '''Gather all the visualization objects
    '''

    def _get_vis3d_list(self):
        return np.array([viz3d.vis3d for viz3d in self.viz3d_list])
<<<<<<< HEAD
<<<<<<< master
#        return np.unique(np.array([viz3d.vis3d for viz3d in self.viz3d_list]))
=======
=======
>>>>>>> bypass2
        # return np.unique(np.array([viz3d.vis3d for viz3d in
        # self.viz3d_list]))

    selected_viz3d = Instance(Viz3D)
<<<<<<< HEAD
>>>>>>> Transformed to python 3
=======
>>>>>>> bypass2

    vot_min = Float(0.0)
    vot_max = Float(1.0)

    vot = Float(0.0)
    '''Visual object time - specifying the current appearance state of an object
    in the television.
    '''
    @on_trait_change('vot')
    def _broadcast_vot(self):
        for vis4d in self.vis3d_list:
            vis4d.vot = self.vot

    vot_slider = Float(0.0)
    '''Visual object time - specifying the current appearance state of an object
    in the television.
    '''
    @on_trait_change('vot_slider')
    def _update_vot_slider(self):
        self.update(self.vot_slider)

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
<<<<<<< HEAD
<<<<<<< master
<<<<<<< master
<<<<<<< HEAD
            raise IndexError('No vizualization module inserted'
                             'into fold television')
=======
            raise IndexError('No vizualization module inserted' \
                'into fold television')
>>>>>>> 2to3
=======
            raise IndexError('No vizualization module inserted' \
                'into fold television')
>>>>>>> interim stage 1
=======
            raise IndexError('No vizualization module inserted'
                             'into fold television')
>>>>>>> Transformed to python 3
=======
            raise IndexError('No vizualization module inserted'
                             'into fold television')
>>>>>>> bypass2
        for viz3d in self.viz3d_list:
            x_min, x_max = viz3d.min_max
            if x_min != None and x_max != None:
                mm.append([x_min, x_max])

        bnodes = np.array(mm, dtype='float_')
        if len(bnodes) > 0:
            bb_min, bb_max = np.min(bnodes[:, 0, :], axis=0), np.max(
                bnodes[:, 1, :], axis=0)
            return bb_min, bb_max
        else:
            return np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float)

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

    bgcolor = Tuple(1.0, 1.0, 1.0)
    fgcolor = Tuple(0.0, 0.0, 0.0)

    mlab = Property(depends_on='input_change')
    '''Get the mlab handle'''

    def _get_mlab(self):
        return self.scene.mlab

    def add(self, viz3d, order=1, name=None):
        '''Add a new visualization objectk.'''
        viz3d.ftv = self
        vis3d = viz3d.vis3d
        if name == None:
            name = viz3d.label
        label = '%s[%s:%s]-%s' % (name,
                                  str(vis3d.__class__),
                                  str(viz3d.__class__),
                                  vis3d
                                  )
        if label in self.viz3d_dict:
            raise KeyError('viz3d object named %s already registered' % label)
        viz3d.order = order
        self.viz3d_dict[label] = viz3d

    viz3d_list = Property

    def _get_viz3d_list(self):
        map_order_viz3d = {}
        for idx, (viz3d) in enumerate(self.viz3d_dict.values()):
            order = viz3d.order
            map_order_viz3d['%5g%5g' % (order, idx)] = viz3d
        return [map_order_viz3d[key] for key in sorted(map_order_viz3d.keys())]

    scene = Instance(MlabSceneModel)
    '''Scene to plot the vizualization pipeline
    '''

    def _scene_default(self):
        return MlabSceneModel()

    label_on = Bool(False)

    def plot(self):
        '''Plot the current visualization objects.
        '''
        fig = self.mlab.gcf()
        bgcolor = tuple(self.bgcolor)
        fgcolor = tuple(self.fgcolor)
        self.mlab.figure(fig, bgcolor=bgcolor, fgcolor=fgcolor)
        for viz3d in self.viz3d_list:
            viz3d.plot()
        if self.label_on:
            oricreate_mlab_label(self.mlab)

    def update(self, vot=0.0, anim_t=0.0, force=False):
        '''Update current visualization.
        '''
        self.vot = vot
        fig = self.mlab.gcf()
        fig.scene.disable_render = True
        for viz3d in self.viz3d_list:
            if force:
                viz3d.vis3d_changed = True
            viz3d.update_t(anim_t)
        fig.scene.disable_render = False

    def show(self, *args, **kw):
        '''Render the visualization.
        '''
        self.mlab.show(*args, **kw)

    traits_view = View(
        VGroup(
            HSplit(
                VGroup(
                    UItem(name='scene',
                          editor=SceneEditor(scene_class=MayaviScene),
                          resizable=True,
                          springy=True,
                          height=800,
                          width=1000),
                ),
                VGroup(
                    VSplit(
                        Item('viz3d_list@', editor=viz3d_dict_editor,
                             show_label=False, width=200),
                        Item('selected_viz3d@', show_label=False,
                             width=100),
                        show_border=True,
                    ),
                ),
            ),
            UItem(
                'vot_slider',
                editor=RangeEditor(
                    low_name='vot_min',
                    high_name='vot_max',
                    format='(%s)',
                    auto_set=True,
                    enter_set=False,
                ),
                resizable=False, height=40, springy=False,
            ),
        ),
        resizable=True,
        width=1.0,
        height=1.0,
        kind='subpanel',
        title='Forming task viewer',
    )


FTV = FormingTaskView3D

if __name__ == '__main__':
    from .point_cloud_viz3d import PointCloud
    ftv = FTV()
    pc = PointCloud()
    ftv.add(pc.viz3d['default'])
    ftv.mlab.options.offscreen = False
    ftv.plot()
    ftv.viz3d_list
    # ftv.mlab.savefig('/home/rch/testing.jpg')
    ftv.configure_traits()
