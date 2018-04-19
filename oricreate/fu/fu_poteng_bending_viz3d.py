#-------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#

import numpy as np
from oricreate.viz3d import Viz3D
import traits.api as t


class LineValueVi3D(Viz3D):

    glyph_source = t.Enum('cone_source')

    def plot(self):
        m = self.ftv.mlab
        x, y, z, u, v, w, scalars, scale_factor = self.get_values()
        vectors = m.quiver3d(
            x, y, z, u, v, w,
            name='Bending moments'
        )
        vectors.glyph.glyph_source.glyph_source = \
            vectors.glyph.glyph_source.glyph_dict[self.glyph_source]

        vectors.glyph.glyph.scale_factor = scale_factor
        vectors.glyph.glyph.clamping = False
        vectors.glyph.color_mode = 'color_by_scalar'
        vectors.glyph.glyph_source.glyph_source.center = np.array(
            [0.,  0.,  0.])
        ds = vectors.mlab_source.dataset
        ds.point_data.scalars = scalars
        ds.point_data.scalars.name = 'bending moments'
        ds.point_data.vectors = np.c_[u, v, w]
        ds.point_data.vectors.name = 'directions'

        self.pipes['vectors'] = vectors

    def update(self, vot=0.0):
        x, y, z, u, v, w, scalars, scale_factor = self.get_values()
        vectors = self.pipes['vectors']
        vectors.mlab_source.set(x=x, y=y, z=z, u=u, v=v, w=w)
        vectors.mlab_source.set(scalars=scalars)
        vectors.mlab_source.set(vectors=np.c_[u, v, w])
        vectors.glyph.glyph.scale_factor = scale_factor

        lut = vectors.module_manager.scalar_lut_manager
        lut.set(
            show_scalar_bar=True,
            show_legend=True,
            data_name='moment'
        )


class FuPotEngBendingViz3D(LineValueVi3D):
    '''Visualize the bending modments
    '''

    def get_values(self):
        fu_tot_poteng = self.vis3d
        ft = fu_tot_poteng.forming_task
        cp = ft.formed_object
        L_lengths = cp.L_lengths
        L_scale = np.average(L_lengths)

        m_u = fu_tot_poteng.m_u

        sim_hist = ft.sim_history
        sim_hist.vot = self.vis3d.vot
        cp.x_0 = np.copy(sim_hist.x_t[0])
        cp.u = np.copy(sim_hist.u)

        iL_phi = cp.iL_psi - cp.iL_psi_0

        iL_m = fu_tot_poteng.kappa * iL_phi
        max_m, min_m = np.max(iL_m), np.min(iL_m)
        delta_m = max_m - min_m
        print('delta_m', delta_m)
        if delta_m != 0:
            print('-1.0', -1. / delta_m)
            print(L_scale)
            scale_factor = -1. / delta_m * L_scale
        else:
            scale_factor = -1.

        norm_F_normals = cp.F_normals
        iL_norm_F_normals = norm_F_normals[cp.iL_F]
        n0 = iL_norm_F_normals[:, 0, :]
        n1 = iL_norm_F_normals[:, 1, :]
        n01 = n0 + n1

        norm_n01 = np.linalg.norm(n01, axis=1)
        normed_n01 = n01 / norm_n01[:, np.newaxis]

        m_vector = iL_m[:, np.newaxis] * normed_n01

        iL_N = cp.L[cp.iL]

        x_t = cp.x_0 + cp.u
        iL_x0 = x_t[iL_N[:, 0]]
        iL_x1 = x_t[iL_N[:, 1]]
        L_ref = iL_x0 + 0.5 * (iL_x1 - iL_x0)

        x, y, z = L_ref.reshape(-1, 3).T
        u, v, w = m_vector.reshape(-1, 3).T

        return x, y, z, u, v, w, iL_m / m_u, scale_factor
