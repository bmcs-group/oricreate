# -------------------------------------------------------------------------
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
# Created on Nov 18, 2011 by: matthias

from traits.api import \
    implements, Array

from fu import \
    Fu
import numpy as np
from oricreate.opt import \
    IFu
from oricreate.viz3d import \
    Visual3D


class FuNodeDist(Fu, Visual3D):

    '''Optimization criteria based on the distance between specified nodes.
    '''

    implements(IFu)

    L = Array(int)

    def get_f(self, t=0):
        '''Get the the norm of distances between the individual nodes.

        ...math::
            d_l = ld(1,x_{Ie}) - ld(2,x_{Ie})


        '''
        cp = self.forming_task.formed_object
        u = cp.u
        x = self.forming_task.x_0 + u
        L = self.L
        v_arr = x[L[:, 1], :] - x[L[:, 0], :]
        l_arr = np.sqrt(np.sum(v_arr ** 2, axis=1))
        return np.sum(l_arr)

    def get_f_du(self, t=0):
        r'''Get the derivatives of the distance vectors with respect 
        to the displacements.

        '''
        cp = self.forming_task.formed_object
        u = cp.u
        x = self.forming_task.x_0 + u
        L = self.L
        v_arr = x[L[:, 1], :] - x[L[:, 0], :]
        l_arr = np.sqrt(np.sum(v_arr ** 2, axis=1))
        L_total = np.sum(l_arr)

        I = L[:, 0]
        J = L[:, 1]

        x_I = x[I]
        x_J = x[J]

        f_du_I = -(x_J - x_I) / L_total
        f_du_J = (x_J - x_I) / L_total

        f_du = np.zeros(
            (cp.n_N, cp.n_D), dtype='float_')

        if L.size > 0:
            f_du[I, :] += f_du_I
            f_du[J, :] += f_du_J

        f_du = f_du.flatten()
        return f_du
