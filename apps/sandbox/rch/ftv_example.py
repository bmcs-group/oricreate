'''

'''

from traits.api import \
    Tuple, Property
from traitsui.api import \
    View, HSplit, VGroup, Item, UItem, TableEditor, ObjectColumn

import numpy as np
from oricreate.api import FTV, Vis3D, Viz3D


viz3d_dict_editor = TableEditor(
    columns=[ObjectColumn(label='Label', name='label', editable=False),
             ObjectColumn(label='Start', name='anim_t_start', editable=False),
             ObjectColumn(label='End', name='anim_t_end', editable=False),
             ],
    selection_mode='row',
    selected='object.selected_viz3d',
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


class PointCloudShiftedViz3D(PointCloudViz3D):

    def update(self):
        x, y, z, s = self.vis3d.points_t
        points = np.c_[x + 2.0, y, z]
        self.pipes['points'].mlab_source.set(points=points, scalars=s)


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
                         something_else=PointCloudShiftedViz3D)


ftv = FTV()
pc = PointCloud()
ftv.add(pc.viz3d['default'])
ftv.add(pc.viz3d['something_else'])
ftv.plot()

ftv.configure_traits()

ftv.plot()
ftv.update(vot=0.9)
ftv.update(vot=0.8)
ftv.show()
