'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    Property, Bool, Int, cached_property, on_trait_change
import numpy as np
from oricreate.viz3d import Viz3D


class FuTargetFaceViz3D(Viz3D):
    '''Visualize the target face
    '''

    label = 'target face'

    show_ff_pipe = Bool(True, input=True)
    show_ff_nodes = Bool(False, input=True)

    # constrain opacity
    opacity_min = Int(0)
    opacity_max = Int(100)

    opacity = Int(20, input=True)

    def plot(self):

        vot = self.vis3d.vot
        tf = self.vis3d.control_face
        m = self.ftv.mlab
        x, y, z = self.ftv.xyz_grid
        tf_pipe = m.contour3d(x, y, z,
                              lambda x, y, z: tf.Rf(x, y, z, vot),
                              contours=[0.0])
        tf_pipe.visible = self.show_ff_pipe
        tf_pipe.module_manager.scalar_lut_manager.lut.table = self.lut
        self.pipes['tf_pipe'] = tf_pipe

#         x, y, z = self.x_t[0][self.nodes_id].T
#         sf = self.scale_factor * 0.5
#         ff_nodes = self.scene.mlab.points3d(x, y, z,
#                                             scale_factor=sf,
#                                             color=(0.5, 0., 0.))
#         ff_nodes.visible = self.show_ff_nodes
#         return ff_nodes

    def update(self, vot=0.0):
        if self.show_ff_pipe:
            vot = self.vis3d.vot
            tf = self.vis3d.control_face
            x, y, z = self.ftv.xyz_grid
            Rf = tf.Rf(x, y, z, vot)
            self.pipes['tf_pipe'].mlab_source.set(scalars=Rf)

        if self.show_ff_nodes:
            x, y, z = self.x_t[self.fold_step][self.nodes_id].T
            self.ff_nodes.mlab_source.reset(x=x, y=y, z=z)

    # constrain colormap
    lut = Property(depends_on='opacity')

    @cached_property
    def _get_lut(self):
        lut = np.zeros((256, 4), dtype=Int)
        alpha = 255 * self.opacity / 100
        lut[:] = np.array([0, 0, 255, int(round(alpha))], dtype=Int)
        return lut

    @on_trait_change('show_ff_pipe, lut')
    def update_ff_pipe_vis(self):
        self.ff_pipe.module_manager.scalar_lut_manager.lut.table = self.lut
        self.ff_pipe.visible = self.show_ff_pipe

    @on_trait_change('show_ff_nodes')
    def update_ff_nodes_vis(self):
        self.ff_nodes.visible = self.show_ff_nodes


class FuTargetFacesViz3D(Viz3D):
    '''List of target faces.
    '''
    label = 'target_faces'
