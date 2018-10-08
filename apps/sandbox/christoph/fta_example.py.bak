'''
Created on 23.01.2017
'''
import numpy as np
from traits.api import Tuple, Property, cached_property
from traitsui.api import TableEditor, ObjectColumn
from oricreate.api import FTV, Viz3D, Vis3D
from apps.sandbox.christoph.forming_task_anim3d import FTA

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
        points = np.c_[x + 2, y, z]
        self.pipes['points'].mlab_source.set(points=points, scalars=s)


class PointCloud(Vis3D):
    '''State object
    '''
    p = Tuple
    '''Point positions
    '''

    def _p_default(self):
        x, y, z, s = np.random.random((4, 100))
        return x, y, z, s

    points_t = Property(Tuple)

    def _get_points_t(self):
        x, y, z, s = self.p
        c = self.vot
        return x * c, y * c, z * c, s * c

    viz3d_classes = dict(default=PointCloudViz3D,
                         something_else=PointCloudShiftedViz3D)

if __name__ == '__main__':

    pc = PointCloud(anim_t_start=0, anim_t_end=40)
    ftv = FTV()
    ftv.add(pc.viz3d['default'])
#     ftv.add(pc.viz3d['something_else'])

    fta = FTA(ftv=ftv)
    fta.init_view(a=0, e=0, d=8, f=(0, 0, 0), r=0)
#     fta.add_cam_move(e=50, n=20,
#                      duration=10,
#                      azimuth_move='linear',
#                      elevation_move='damped',
#                      distance_move='damped')
#     fta.add_cam_move(e=0, n=20,
#                      duration=30,
#                      azimuth_move='damped',
#                      elevation_move='damped',
#                      distance_move='damped')
    fta.plot()
    fta.configure_traits()
