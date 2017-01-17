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
# Created on Sep 7, 2011 by: rch

from oricreate.crease_pattern import \
    CreasePatternState
from oricreate.forming_tasks import \
    FactoryTask
from traits.api import \
    Float, Int, Property, cached_property, Callable

import numpy as np
import sympy as sp

x_, y_ = sp.symbols('x, y')


class HPCPFactory(FactoryTask):

    '''Generate a Yoshimura crease pattern based
    on the specification of its parameters.

    .. todo::
       Default parameters do not work. d_x not described.
    '''

    L_x = Float(4, geometry=True)
    L_y = Float(2, geometry=True)

    n_stripes = Int(8, geometry=True)

    def deliver(self):
        return CreasePatternState(X=self.X,
                                  L=self.L,
                                  F=self.F)

    X = Property

    def _get_X(self):
        return self._geometry[0]

    L = Property

    def _get_L(self):
        return self._geometry[1]

    F = Property

    def _get_F(self):
        return self._geometry[2]

    geo_transform = Callable

    def _geo_transform_default(self):
        return lambda X_arr: X_arr

    _geometry = Property(depends_on='+geometry')

    @cached_property
    def _get__geometry(self):

        n_x = (self.n_stripes + 1) * 2 + 1

        L_x = self.L_x
        L_y = self.L_y

        x_e, y_e = np.mgrid[0:L_x:complex(n_x), 0:L_y:complex(n_x)]

        x = np.c_[x_e.flatten(), y_e.flatten(), np.zeros_like(x_e.flatten())]

        N = np.arange(n_x * n_x).reshape(n_x, n_x)

        f11 = np.array([N[:-1, :-2], N[1:, 1:-1], N[:-1, 2:]])
        f11 = np.einsum('nij->ijn', f11)

        f22 = np.array([N[1:, :-2], N[1:, 2:], N[:-1, 1:-1]])
        f22 = np.einsum('nij->ijn', f22)

        f33 = np.array([N[:-2, :-1], N[2:, :-1], N[1:-1, 1:]])
        f33 = np.einsum('nij->ijn', f33)

        f4 = np.array([N[:-2, 1:], N[2:, 1:], N[1:-1, :-1]])
        f44 = np.einsum('nij->ijn', f4)

        def get_triangle(arr):
            '''Given a twodimensional array return
            the left triangle
            '''
            n_i, n_j = arr.shape[0], arr.shape[1]
            i, j = np.mgrid[:n_i, :n_j]
            i_arr, j_arr = np.where((i <= j) & (i <= (n_j - 1 - j)))
            return arr[i_arr, j_arr]

        FL = np.vstack([get_triangle(f11[::2, ::2])[:-1],
                        get_triangle(f11[1:-1:2, 1:-1:2]),
                        get_triangle(f22[::2, 1:-1:2]),
                        get_triangle(f22[1:-1:2, 2:-2:2])]
                       )

        FR = np.vstack([get_triangle(f22[-1::-2, ::2])[:-1],
                        get_triangle(f22[-2:1:-2, 1:-1:2]),
                        get_triangle(f11[-1::-2, 1:-1:2]),
                        get_triangle(f11[-2:1:-2, 2:-2:2])]
                       )

        FB = np.vstack([get_triangle(f33[::2, ::2].swapaxes(0, 1))[:-1],
                        get_triangle(f33[1:-1:2, 1:-1:2].swapaxes(0, 1)),
                        get_triangle(f44[1:-1:2, ::2].swapaxes(0, 1)),
                        get_triangle(f44[2:-2:2, 1:-1:2].swapaxes(0, 1))]
                       )

        FT = np.vstack([get_triangle(f44[::2, -1::-2].swapaxes(0, 1))[:-1],
                        get_triangle(f44[1:-1:2, -2:1:-2].swapaxes(0, 1)),
                        get_triangle(f33[1:-1:2, -1::-2].swapaxes(0, 1)),
                        get_triangle(f33[2:-2:2, -2:1:-2].swapaxes(0, 1))]
                       )

        facets = np.vstack([FL, FR, FB, FT])

        # identify lines
        ix_arr = np.array([[0, 1], [1, 2], [2, 0]])
        L_N = facets[:, ix_arr].reshape(-1, 2)
        n_L = len(L_N)
        n_N = len(x)
        NN_L = np.zeros((n_N, n_N), dtype='int') - 1
        NN_L[L_N[:, 0], L_N[:, 1]] = np.arange(n_L)
        NN_L[L_N[:, 1], L_N[:, 0]] = np.arange(n_L)
        i, j = np.mgrid[:n_N, :n_N]
        i_arr, j_arr = np.where((i > j) & (NN_L > -1))
        l_arr = NN_L[i_arr, j_arr]
        lines = L_N[l_arr]

        # shrink / condense
        N_connected = np.where(np.sum(NN_L + 1, axis=1) > 0)[0]
        N_enum = np.zeros(n_N, dtype=np.int_) - 1
        N_enum[N_connected] = np.arange(len(N_connected))

        x_red = x[N_connected, :]
        l_red = N_enum[lines]
        f_red = N_enum[facets]

        x_red = self.geo_transform(x_red)

        return (x_red, l_red, f_red,)

if __name__ == '__main__':

    def geo_transform(x_arr):
        alpha = np.pi / 8.0
        x_max = np.max(x_arr, axis=0)
        x_min = np.min(x_arr, axis=0)
        T = (x_max - x_min) / 2.0
        x_arr -= T[np.newaxis, :]

        R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]], dtype=np.float_)
        x_rot = np.einsum('ij,nj->ni', R, x_arr)
        return x_rot

    yf = HPCPFactory(L_x=30,
                     L_y=30,
                     n_stripes=2,
                     geo_transform=geo_transform
                     )

    cp = yf.formed_object

    print 'free dofs =', cp.n_dofs - cp.n_L - 6

    import pylab as p
    cp.plot_mpl(p.axes(), nodes=False, lines=False, facets=True)
    p.show()
