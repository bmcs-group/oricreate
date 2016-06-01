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


class FuBendingEnergyViz3D(Viz3D):
    '''Visualize the crease Pattern
    '''

    def get_values(self):
        fu_tot_poteng = self.vis3d
        ft = fu_tot_poteng.forming_task
        cp = ft.formed_object

        iL_phi = cp.iL_psi2 - cp.iL_psi_0
        iL_length = np.linalg.norm(cp.iL_vectors, axis=1)
        iL_m = fu_tot_poteng.kappa * iL_phi * iL_length
        print 'FuPotengVerify'
        print 'kappa', fu_tot_poteng.kappa
        print 'iL_phi', iL_phi
        print 'iL_length', iL_length
        print 'iL_m', iL_m
        max_m = np.max(iL_m)
        iL_m /= max_m

        norm_F_normals = cp.F_normals
        iL_norm_F_normals = norm_F_normals[cp.iL_F]
        n0 = iL_norm_F_normals[:, 0, :]
        n1 = iL_norm_F_normals[:, 1, :]
        n01 = n0 + n1

        norm_n01 = np.linalg.norm(n01, axis=1)
        normed_n01 = n01 / norm_n01[:, np.newaxis]

        m_vector = iL_m[:, np.newaxis] * normed_n01

        iL_N = cp.L[cp.iL]

        iL_x0 = cp.x[iL_N[:, 0]]
        iL_x1 = cp.x[iL_N[:, 1]]
        L_ref = iL_x0 + 0.5 * (iL_x1 - iL_x0)

        x, y, z = L_ref.reshape(-1, 3).T
        u, v, w = m_vector.reshape(-1, 3).T

        return x, y, z, u, v, w

    def plot(self):

        m = self.ftv.mlab
        x, y, z, u, v, w = self.get_values()
        self.quifer3d_pipe = m.quiver3d(x, y, z, u, v, w)

    def update(self):
        x, y, z, u, v, w = self.get_values()
        self.quifer3d_pipe.mlab_source.set(x=x, y=y, z=z, u=u, v=v, w=w)
