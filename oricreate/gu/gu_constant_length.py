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

from traits.api import \
    implements
from gu import \
    Gu, IGu
import numpy as np


class GuConstantLength(Gu):

    '''Constant length constraint.
    '''
    implements(IGu)

    def get_G(self, t=0.0):
        '''Calculate the residue for constant crease length
        given the fold vector dX.
        '''
        cp = self.forming_task.formed_object
        v_0 = cp.L_vectors
        u = cp.u
        u_i, u_j = u[cp.L.T]
        v_u_i = np.sum(v_0 * u_i, axis=1)
        v_u_j = np.sum(v_0 * u_j, axis=1)
        u_ij = np.sum(u_i * u_j, axis=1)
        u_ii = np.sum(u_i ** 2, axis=1)
        u_jj = np.sum(u_j ** 2, axis=1)
        G = 2 * v_u_j - 2 * v_u_i - 2 * u_ij + u_ii + u_jj

        return G

    def get_G_du(self, t=0.0):
        '''Calculate the residue for constant crease length
        given the fold vector dX.
        '''
        cp = self.forming_task.formed_object
        G_du = np.zeros((cp.n_L, cp.n_N, cp.n_D), dtype='float_')

        # running crease line index
        if cp.n_L > 0:
            v_0 = cp.L_vectors
            u = cp.u
            i, j = cp.L.T
            u_i, u_j = u[cp.L.T]
            l = np.arange(cp.n_L)
            G_du[l, i, :] += -2 * v_0 + 2 * u_i - 2 * u_j
            G_du[l, j, :] += 2 * v_0 - 2 * u_i + 2 * u_j

        # reshape the 3D matrix to a 2D matrix
        # with rows for crease lines and columns representing
        # the derivatives with respect to the node displacements
        # in 3d.
        #
        G_du = G_du.reshape(cp.n_L, cp.n_N * cp.n_D)
        return G_du

if __name__ == '__main__':

    from oricreate.api import CreasePatternState, CustomCPFactory

    cp = CreasePatternState(X=[[-4, -5, -3],
                               [0, 0.0, 0],
                               [1.0, 0.1, 0],
                               ],
                            L=[[0, 1], [1, 2], [2, 0]],
                            )

    forming_task = CustomCPFactory(formed_object=cp)
    constant_length = GuConstantLength(forming_task)

    U = np.zeros_like(cp.X)
    U[2] += 1.0

    print [constant_length.get_G(U, 0)]
    print [constant_length.get_G_du(U, 0)]
