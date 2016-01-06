r'''

Produce a video with photo stickers on the facets
-------------------------------------------------

Code structure
==============
@todo: who defines the viz objects - viz objects are transferring the state of a vis object into the 
       supplied visualization format - mlab object. A viz object can serve several differnt vis objects.
@todo: cleanup the euler angles.

Visualization
=============
@todo: provide viz objects also for operators - viz bases, viz normals (done) - and constraints 
       - dofconstraints, -targetsurface
@todo: define the mapping objects for the image stickers specifying the covered facets
@todo: generalize the simulation time loop - update constraints for time ``t``, get 
        the solution and store it in the simulation history
@todo: include camera path into the animation loop.
@todo: provide a separate animator class with CameraStations and CameraTransitions.  

'''

from traits.api import \
    Instance, Property, cached_property, \
    File, Array, List, Float
import numpy as np
from oricreate.api import \
    FTV, FTA
from oricreate.crease_pattern import \
    CreasePatternNormalsViz3D, CreasePatternBasesViz3D
from oricreate.crease_pattern.crease_pattern_viz3d import \
    CreasePatternThickViz3D
from oricreate.viz3d.viz3d import Viz3D
from sim03_eft_logo_viz3d import \
    FacetsWithImageViz3D
from sim_task_twist_folding import \
    TwistFolding


def oricreate_mlab_label(m):
    atext_width = 0.18
    m.text(1 - atext_width, 0.003, 'oricreate',
           color=(.7, .7, .7), width=atext_width)


class FoldedStickersFTV(FTV):

    sim_task = Instance(TwistFolding)

    def _twist_folding_default(self):
        return TwistFolding()

    fname_eft_front = File('family_small.png')

    fname_eft_back = File('family_back_small.png')

    back_offset = Float(0.02)

    front_offset = Float(0.02)

    tube_radius = Float(0.01)

    plane_offsets = Array(value=[0.2, -0.20], dtype='float_')

    F_ref = Array(value=[0, 2, 4, 6, 8, 16, 1, 12, 14, ], dtype='int_')

    N_ref = Array(value=[0, 3, 15, 12, 7, 4, 1, 11, 13], dtype='int_')

    F_covered = Array(value=[[0, 9], [2, 11], [4, 13], [6, 15],
                             [8, 17], [16, 7], [1, 10], [12, 3], [14, 5]], dtype='int_')

    atimes = Array(
        value=[90.0, 180.0, -90, 0, 45, 90, 90.0, -90.0, 0.0], dtype='float_')

    im_widths = Array(value=[4, 4, 4, 4, 4, 4, 4, 4, 4], dtype='float_')

    imfiles_front = List

    def _imfiles_front_default(self):
        return [self.fname_eft_front for i in range(len(self.F_ref))]

    imfiles_back = List

    def _imfiles_back_default(self):
        return [self.fname_eft_back for i in range(len(self.F_ref))]

    im_front_offsets = List

    def _im_front_offsets_default(self):
        return [
            [0, 0, self.front_offset],
            [0, 0, self.front_offset],
            [0, 0, self.front_offset],
            [0, 0, self.front_offset],
            [-self.shift_l, -self.shift_l, self.front_offset],
            [0, -1, self.front_offset],
            [-2, 0, self.front_offset],
            [0, -1, self.front_offset],
            [0, -1, self.front_offset]
        ]

    im_back_offsets = List

    def _im_back_offsets_default(self):
        return [
            [0, 0, -self.back_offset],
            [0, 0, -self.back_offset],
            [0, 0, -self.back_offset],
            [0, 0, -self.back_offset],
            [-self.shift_l, -self.shift_l, -self.back_offset],
            [0, -1, -self.back_offset],
            [-2, 0, -self.back_offset],
            [0, -1, -self.back_offset],
            [0, -1, -self.back_offset]
        ]

    edge_len = Float(2.0)

    x_offset = Float(1.0)

    shift_x = Property

    def _get_shift_x(self):
        c45 = np.cos(np.pi / 4)
        return self.x_offset + self.edge_len / c45 - self.edge_len

    shift_l = Property

    def _get_shift_l(self):
        c45 = np.cos(np.pi / 4)
        return self.shift_x * c45

    back_viz3d = Property
    '''Sticker for the back side.
    '''
    @cached_property
    def _get_back_viz3d(self):
        back_viz3d = FacetsWithImageViz3D(
            label='EFT back',
            vis3d=self.sim_task.sim_history,
            F_ref=self.F_ref,  # + F_ref,
            N_ref=self.N_ref,  # + N_ref,
            F_covered=self.F_covered,  # + F_covered,
            atimes=self.atimes,  # + atimes,
            im_files=self.imfiles_back,  # imfiles_front +
            im_widths=self.im_widths,  # + im_widths,
            im_offsets=self.im_back_offsets,  # im_front_offsets +
        )
        return back_viz3d

    front_viz3d = Property

    @cached_property
    def _get_front_viz3d(self):
        front_viz3d = FacetsWithImageViz3D(
            label='EFT front',
            vis3d=self.sim_task.sim_history,
            F_ref=self.F_ref,
            N_ref=self.N_ref,
            F_covered=self.F_covered,
            atimes=self.atimes,
            im_files=self.imfiles_front,
            im_widths=self.im_widths,
            im_offsets=self.im_front_offsets
        )
        return front_viz3d

    eftlogo_normals = Instance(Viz3D)

    def _eftlogo_normals_default(self):
        return CreasePatternNormalsViz3D(label='EFT normals', vis3d=self.sim_task.sim_history)

    eftlogo_bases = Instance(Viz3D)

    def _eftlogo_bases_default(self):
        return CreasePatternBasesViz3D(label='EFT bases', vis3d=self.sim_task.sim_history)

    eft_thick_viz3d = Instance(Viz3D)

    def _eft_thick_viz3d_default(self):
        return CreasePatternThickViz3D(label='EFT thick', vis3d=self.sim_task.sim_history,
                                       plane_offsets=self.plane_offsets)

    def plot(self):
        vis3d = self.sim_task.sim_history
        vis3d.viz3d.set(tube_radius=self.tube_radius)
        # Configure scene
        self.add(vis3d.viz3d)
#         self.add(self.eft_thick_viz3d)
        self.add(self.front_viz3d)
        self.add(self.back_viz3d)
        # self.add(self.labels_viz3d)
        # self.add(self.eftlogo_normals)
        # self.add(self.eftlogo_bases)
        m = self.mlab
        super(FoldedStickersFTV, self).plot()
        oricreate_mlab_label(m)

if __name__ == '__main__':
    twist_folding = TwistFolding(n_u=40)
    ftv = FoldedStickersFTV(sim_task=twist_folding)
    fta = FTA(ftv=ftv)
    fta.init_view(a=110, e=35, d=11, f=(0, 0, 0), r=160)
    fta.add_cam_move(a=90, e=10, n=50, d=7, r=180,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(0, 1, 40),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')
    fta.add_cam_move(d=5, n=10,
                     duration=10,
                     vot_fn=lambda cmt: np.ones_like(cmt),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')
    fta.add_cam_move(a=0, e=160, d=8, n=25,
                     duration=10,
                     vot_fn=lambda cmt: np.ones_like(cmt),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')
    fta.add_cam_move(e=170, r=180 + 45, d=12, n=30,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(1, 0.2, 30),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped',
                     roll_move='damped')
    fta.add_cam_move(r=180 + 45 + 90, n=15,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(0.2, 0.0, 15),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped',
                     roll_move='damped')
    fta.add_cam_move(r=180 + 45 + 180, n=15,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(0.0, 0.1, 15),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped',
                     roll_move='damped')
    fta.add_cam_move(r=180 + 45 + 270, n=15,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(0.1, 0.2, 15),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped',
                     roll_move='damped')
    fta.add_cam_move(a=110, e=35, d=11, r=160, n=60,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(0.2, 0.0, 60),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped',
                     roll_move='damped')

    fta.plot()
    fta.render()
    fta.configure_traits()
