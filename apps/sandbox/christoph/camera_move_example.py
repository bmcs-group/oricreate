<<<<<<< master
'''
Created on 25.01.2017

@author: cthoennessen
'''
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from oricreate.viz3d.forming_task_anim3d import CamStation, CamMove
from traits.api import HasStrictTraits, Instance, Property, Range, Button
from traitsui.api import View, VGroup, UItem, HGroup
from tvtk.pyface.scene_editor import SceneEditor


class CameraMovementExample(HasStrictTraits):

    scene = Instance(MlabSceneModel, ())
    mlab = Property(depends_on='input_change')

    b1 = Button(label='From')
    b2 = Button(label='To')
    b_pv = Button(label='Print View')

    cam_move = Instance(CamMove, ())
    n_t = 101
    cur_frame = Range(0, n_t - 1)

    def _get_mlab(self):
        return self.scene.mlab

    def plot(self, x, y, z, s):
        self.mlab.barchart(x, y, z, s, colormap='Spectral')

    def _b1_fired(self):
        '''Set the from_station of the cam_move
        '''
        v = self.mlab.view()
        r = self.mlab.roll()
        self.cam_move.from_station = CamStation(azimuth=v[0],
                                                elevation=v[1],
                                                distance=v[2],
                                                focal_point=tuple(v[3]),
                                                roll=r)
        print('from_station set.')

    def _b2_fired(self):
        '''Set the to_station of the cam_move
        '''
        v = self.mlab.view()
        r = self.mlab.roll()
        self.cam_move.to_station = CamStation(azimuth=v[0],
                                              elevation=v[1],
                                              distance=v[2],
                                              focal_point=tuple(v[3]),
                                              roll=r)
        print('to_station set.')

    def _cur_frame_changed(self):
        '''Change to the selected frame
        '''
        self.cam_move.n_t = self.n_t
        trans_arr = self.cam_move.transition_arr
        a = trans_arr[0][self.cur_frame]
        e = trans_arr[1][self.cur_frame]
        d = trans_arr[2][self.cur_frame]
        f = trans_arr[3][self.cur_frame]
        r = trans_arr[4][self.cur_frame]
#         self.mlab.view(a, e, d, f, r) # with roll
        self.mlab.view(a, e, d, f)  # without roll

    def _b_pv_fired(self):
        '''Print current view
        '''
        print(self.mlab.view(), self.mlab.roll())

    traits_view = View(
        VGroup(
            UItem('scene',
                  editor=SceneEditor(scene_class=MayaviScene)
                  ),
            UItem('cur_frame'),
            HGroup(
                UItem('b1'),
                UItem('b2'),
                UItem('b_pv')
            )
        ),
        resizable=True
    )


if __name__ == '__main__':

    x, y, z, s = [-1, 0, 1], [-1, 0, 1], [0, 0, 0], [1, 3, 2]
    cme = CameraMovementExample()
    cme.plot(x, y, z, s)
    cme.configure_traits()
=======
'''
Created on 25.01.2017

@author: cthoennessen
'''
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from oricreate.viz3d.forming_task_anim3d import CamStation, CamMove
from traits.api import HasStrictTraits, Instance, Property, Range, Button
from traitsui.api import View, VGroup, UItem, HGroup
from tvtk.pyface.scene_editor import SceneEditor


class CameraMovementExample(HasStrictTraits):

    scene = Instance(MlabSceneModel, ())
    mlab = Property(depends_on='input_change')

    b1 = Button(label='From')
    b2 = Button(label='To')
    b_pv = Button(label='Print View')

    cam_move = Instance(CamMove, ())
    n_t = 101
    cur_frame = Range(0, n_t - 1)

    def _get_mlab(self):
        return self.scene.mlab

    def plot(self, x, y, z, s):
        self.mlab.barchart(x, y, z, s, colormap='Spectral')

    def _b1_fired(self):
        '''Set the from_station of the cam_move
        '''
        v = self.mlab.view()
        r = self.mlab.roll()
        self.cam_move.from_station = CamStation(azimuth=v[0],
                                                elevation=v[1],
                                                distance=v[2],
                                                focal_point=tuple(v[3]),
                                                roll=r)
        print('from_station set.')

    def _b2_fired(self):
        '''Set the to_station of the cam_move
        '''
        v = self.mlab.view()
        r = self.mlab.roll()
        self.cam_move.to_station = CamStation(azimuth=v[0],
                                              elevation=v[1],
                                              distance=v[2],
                                              focal_point=tuple(v[3]),
                                              roll=r)
        print('to_station set.')

    def _cur_frame_changed(self):
        '''Change to the selected frame
        '''
        self.cam_move.n_t = self.n_t
        trans_arr = self.cam_move.transition_arr
        a = trans_arr[0][self.cur_frame]
        e = trans_arr[1][self.cur_frame]
        d = trans_arr[2][self.cur_frame]
        f = trans_arr[3][self.cur_frame]
        r = trans_arr[4][self.cur_frame]
#         self.mlab.view(a, e, d, f, r) # with roll
        self.mlab.view(a, e, d, f)  # without roll

    def _b_pv_fired(self):
        '''Print current view
        '''
        print(self.mlab.view(), self.mlab.roll())

    traits_view = View(
        VGroup(
            UItem('scene',
                  editor=SceneEditor(scene_class=MayaviScene)
                  ),
            UItem('cur_frame'),
            HGroup(
                UItem('b1'),
                UItem('b2'),
                UItem('b_pv')
            )
        ),
        resizable=True
    )


if __name__ == '__main__':

    x, y, z, s = [-1, 0, 1], [-1, 0, 1], [0, 0, 0], [1, 3, 2]
    cme = CameraMovementExample()
    cme.plot(x, y, z, s)
    cme.configure_traits()
>>>>>>> interim stage 1
