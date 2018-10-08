'''
Created on Dec 3, 2015

@author: rch
'''

from traits.api import \
    Array, Tuple, Property, Bool, Float, Color, List

import numpy as np
from oricreate.viz3d import Viz3D
import traitsui.api as tui


class CreasePatternViz3D(Viz3D):
    '''Visualize the crease Pattern
    '''
    N_selection = Array(int)
    L_selection = Array(int)
    F_selection = Array(int)

    lines = Bool(True)

    N_L_F = Property(Tuple)
    '''Geometry with applied selection arrays.
    '''

    def _get_N_L_F(self):
        cp = self.vis3d
        x, L, F = cp.x, cp.L, cp.F
        if len(self.N_selection):
            x = x[self.N_selection]
        if len(self.L_selection):
            L = L[self.L_selection]
        if len(self.F_selection):
            F = F[self.F_selection]
        return x, L, F

    min_max = Property
    '''Rectangular bounding box. 
    '''

    def _get_min_max(self):
        vis3d = self.vis3d
        return np.min(vis3d.x, axis=0), np.max(vis3d.x, axis=0)

    facet_color = Color((0.6, 0.625, 0.683))
#    facet_color = Color((0.0, 0.425, 0.683))
#    facet_color = Color((0.4, 0.4, 0.7))
#    facet_color = Color((0.0 / 255.0, 84.0 / 255.0, 159.0 / 255.0))
#   facet_color = Color((64.0 / 255.0, 127.0 / 255.0, 183.0 / 255.0))
#    facet_color = Color((0.0 / 255.0, 97.0 / 255.0, 101.0 / 255.0))

    def plot(self):

        m = self.ftv.mlab
        N, L, F = self.N_L_F
        x, y, z = N.T
        if len(F) > 0:
            cp_pipe = m.triangular_mesh(x, y, z, F,
                                        line_width=3,
                                        # color=self.facet_color.toTuple()[:-1],
                                        color=(0.6, 0.625, 0.683),
                                        #                                        color=(0.0, 0.425, 0.683),
                                        name='Crease pattern')
#                                        color=self.facet_color.toTuple()[:-1])
            if self.lines is True:
                cp_pipe.mlab_source.dataset.lines = L
                tube = m.pipeline.tube(cp_pipe,
                                       tube_radius=self.tube_radius)
                lines = m.pipeline.surface(tube, color=(0.1, 0.1, 0.1))
                self.pipes['lines'] = lines
        else:
            cp_pipe = m.points3d(x, y, z, scale_factor=0.2)
            cp_pipe.mlab_source.dataset.lines = L
        self.pipes['cp_pipe'] = cp_pipe

    def update(self, vot=0.0):
        N = self.N_L_F[0]
        cp_pipe = self.pipes['cp_pipe']
        cp_pipe.mlab_source.set(points=N)

    def _get_bounding_box(self):
        N = self.N_L_F[0]
        return np.min(N, axis=0), np.max(N, axis=0)

    def _get_max_length(self):
        return np.linalg.norm(self._get_bounding_box())

    tube_radius = Float(0.03)

    line_width_factor = Float(0.0024)

    def _get_line_width(self):
        return self._get_max_length() * self.line_width_factor

    traits_view = tui.View(
        tui.VGroup(
            tui.Include('viz3d_view'),
            tui.UItem('tube_radius'),
            tui.UItem('line_width_factor')
        )
    )

    selection_view = traits_view


class CreasePatternDisplViz3D(CreasePatternViz3D):
    '''Visualize the crease pattern with displacement vectors at the nodes
    '''

    warp_scale_factor = Float(1.0, enter_set=True, auto_set=False)
    '''Displacement warp scale
    '''

    N_L_F = Property(Tuple)
    '''Geometry with applied selection arrays.
    '''

    def _get_N_L_F(self):
        cp = self.vis3d
        x, u, L, F = cp.x_0, cp.u, cp.L, cp.F
        if len(self.N_selection):
            x = x[self.N_selection]
            u = u[self.N_selection]
        if len(self.L_selection):
            L = L[self.L_selection]
        if len(self.F_selection):
            F = F[self.F_selection]
        return x, u, L, F

    def plot(self):

        m = self.ftv.mlab
        x_0, u_, L, F = self.N_L_F
        x, y, z = x_0.T
        u, v, w = u_.T
        if len(F) > 0:
            cp_pipe = m.triangular_mesh(x, y, z, F,
                                        line_width=3,
                                        scalars=w,
                                        color=(0.6, 0.625, 0.683),
                                        name='Crease pattern displacement'
                                        # color=self.facet_color.toTuple()[:-1]
                                        )
            if self.lines is True:
                cp_pipe.mlab_source.dataset.lines = L
                tube = m.pipeline.tube(cp_pipe,
                                       tube_radius=0.003)
                lines = m.pipeline.surface(tube, color=(1.0, 1.0, 1.0))
                self.pipes['lines'] = lines
        else:
            cp_pipe = m.points3d(x, y, z,
                                 color=(0.6, 0.625, 0.683),
                                 scale_factor=0.2,
                                 name='Crease pattern displacement'
                                 )
            cp_pipe.mlab_source.dataset.lines = L
        ds = cp_pipe.mlab_source.dataset
        ds.point_data.vectors = u_
        ds.point_data.vectors.name = 'u'
        warp = m.pipeline.warp_vector(cp_pipe)
        surf = m.pipeline.surface(warp)
        self.pipes['warp'] = warp
        self.pipes['cp_pipe'] = cp_pipe

    def update(self, vot=0.0):
        x_0, u_, L, F = self.N_L_F
        u, v, w = u_.T
        cp_pipe = self.pipes['cp_pipe']
#         warp = self.pipes['warp']
#        warp.filter.scale_factor = self.warp_scale_factor
        cp_pipe.mlab_source.set(points=x_0, scalars=w)
        ds = cp_pipe.mlab_source.dataset
        ds.point_data.vectors = u_


class CreasePatternThickViz3D(CreasePatternViz3D):
    '''Visualize facets as if they had thickness
    '''
    thickness = Float(0.06)
    lines = False

    plane_offsets = Array(float, value=[0])

    def _get_N_L_F(self):
        x, L, F = super(CreasePatternThickViz3D, self)._get_N_L_F()
        cp = self.vis3d
        F_sel = slice(None)
        if len(self.F_selection):
            F_sel = F[self.F_selection]

        norm_F_normals = cp.norm_F_normals[F_sel]
        offsets = norm_F_normals[None, :, :] * \
            self.plane_offsets[:, None, None]
        F_x = x[F]
        F_x_planes = F_x[None, :, :, :] + offsets[:, :, None, :]
        x_planes = F_x_planes.reshape(-1, 3)
        F = np.arange(x_planes.shape[0]).reshape(-1, 3)
        return x_planes, L, F


class CreasePatternNormalsViz3D(Viz3D):
    '''Visualize the crease Pattern
    '''

    def get_values(self):
        cp = self.vis3d

        Fa_r = cp.Fa_r
        Fa_normals = cp.Fa_normals

        x, y, z = Fa_r.reshape(-1, 3).T
        u, v, w = Fa_normals.reshape(-1, 3).T
        return x, y, z, u, v, w

    def plot(self):

        m = self.ftv.mlab
        x, y, z, u, v, w = self.get_values()
        self.pipes['quifer3d_pipe'] = m.quiver3d(x, y, z, u, v, w)

    def update(self, vot=0.0):
        x, y, z, u, v, w = self.get_values()
        quifer3d_pipe = self.pipes['quifer3d_pipe']
        quifer3d_pipe.mlab_source.set(x=x, y=y, z=z, u=u, v=v, w=w)


class CreasePatternBasesViz3D(Viz3D):
    '''Visualize the crease Pattern
    '''
    label = 'bases'

    def get_values(self):
        cp = self.vis3d

        Fa_r = cp.Fa_r
        F_L_bases = cp.F_L_bases[:, 0, :, :]
        return Fa_r.reshape(-1, 3), F_L_bases.reshape(-1, 3, 3)

    def plot(self):

        m = self.ftv.mlab
        Fa_r, F_L_bases = self.get_values()
        args_red = tuple(Fa_r.T) + tuple(F_L_bases[..., 0, :].T)
        args_gre = tuple(Fa_r.T) + tuple(F_L_bases[..., 1, :].T)
        args_blu = tuple(Fa_r.T) + tuple(F_L_bases[..., 2, :].T)
        self.pipes['quifer3d_pipe_red'] = m.quiver3d(
            *args_red, color=(1, 0, 0))
        self.pipes['quifer3d_pipe_gre'] = m.quiver3d(
            *args_gre, color=(0, 1, 0))
        self.pipes['quifer3d_pipe_blu'] = m.quiver3d(
            *args_blu, color=(0, 0, 1))

    def update(self, vot=0.0):
        '''Update positions of the bases.
        '''
        Fa_r, F_L_bases = self.get_values()
        x, y, z = Fa_r.T
        u, v, w = F_L_bases[..., 0, :].T
        quifer3d_pipe_red = self.pipes['quifer3d_pipe_red']
        quifer3d_pipe_red.mlab_source.set(x=x, y=y, z=z, u=u, v=v, w=w)
        u, v, w = F_L_bases[..., 1, :].T
        quifer3d_pipe_gre = self.pipes['quifer3d_pipe_gre']
        quifer3d_pipe_gre.mlab_source.set(x=x, y=y, z=z, u=u, v=v, w=w)
        u, v, w = F_L_bases[..., 2, :].T
        quifer3d_pipe_blu = self.pipes['quifer3d_pipe_blu']
        quifer3d_pipe_blu.mlab_source.set(x=x, y=y, z=z, u=u, v=v, w=w)


class CreasePatternNodeNumbersViz3D(Viz3D):
    '''Visualize the crease Pattern
    '''

    label = 'node numbers'
    scale_factor = Float(1.0, auto_set=False, enter_set=True)
    show_node_index = Bool(True)

    text_entries = Array

    def plot(self):
        '''
        This pipeline comprised the labels for all node Indexes
        '''

        m = self.ftv.mlab
        cp = self.vis3d

        text = np.array([])
        pts = cp.x
        x, y, z = cp.x.T
        for i in range(len(pts)):
            temp_text = m.text3d(
                x[i], y[i], z[i] + 0.2 * self.scale_factor, str(i), scale=0.05)
            temp_text.actor.actor.visibility = int(self.show_node_index)
            text = np.hstack([text, temp_text])
        self.text_entries = text

    def update(self, vot=0.0):
        '''
        Update the labels of nodeindexes (node_index_pipeline)
        '''
        cp = self.vis3d
        x, y, z = cp.x.T
        text_entries = self.text_entries
        for i in range(len(self.text_entries)):
            text_entries[i].actor.actor.visibility = int(
                self.show_node_index)
            text_entries[i].actor.actor.position = np.array(
                [x[i], y[i], z[i] + 0.2 * self.scale_factor])
