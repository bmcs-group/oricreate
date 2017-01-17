'''

'''

from traits.api import \
    Tuple, Property
from traitsui.api import \
    View, HSplit, VGroup, Item, UItem, TableEditor, ObjectColumn

import numpy as np
from oricreate.api import FTV, Vis3D, Viz3D


viz3d_dict_editor = TableEditor(
    columns=[ObjectColumn(label='Start', name='anim_t_start', editable=False),
             ObjectColumn(label='End', name='anim_t_end', editable=False),
             ],
    selection_mode='row',
    selected='object.selected_viz3d',
)

ftv_trait_view = View(
    HSplit(
        VGroup(
            Item('viz3d_dict',
                 style='custom', editor=viz3d_dict_editor,
                 show_label=False, springy=True, width=150),
            Item('selected_viz3d@', show_label=False,
                 springy=True,
                 width=800, height=200),
            show_border=True,
        ),
    ),
    resizable=True,
    height=400,
    kind='subpanel',
    title='Forming task viewer',
)


class PointCloudViz3D(Viz3D):
    '''Visualization object
    '''

    def plot(self):
        x, y, z, s = self.vis3d.p
        self.pipes['points'] = self.ftv.mlab.points3d(x, y, z, s)

    def update(self):
        x, y, z, s = self.vis3d.points_t
        points = np.c_[x, y, z]
        self.pipes['points'].mlab_source.set(points=points, scalars=s)


class PointCloud2Viz3D(PointCloudViz3D):
    pass


class PointCloud(Vis3D):
    '''State object
    '''
    p = Tuple
    '''Point positions
    '''

    def _p_default(self):
        x, y, z, s = np.random.random((4, 100))
        print 's_default', s
        return x, y, z, s

    points_t = Property(Tuple)

    def _get_points_t(self):
        x, y, z, s = self.p
        c = self.vot
        return x * c, y * c, z * c, s * c

    viz3d_classes = dict(default=PointCloudViz3D,
                         something_else=PointCloud2Viz3D)


ftv = FTV()
pc = PointCloud()
ftv.add(pc.viz3d['default'])
ftv.add(pc.viz3d['something_else'])

ftv.configure_traits(view=ftv_trait_view)

ftv.plot()
ftv.update(vot=0.9)
ftv.update(vot=0.8)
ftv.show()
