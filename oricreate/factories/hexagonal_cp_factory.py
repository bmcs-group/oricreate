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

from traits.api import \
    Float, Int, Property, cached_property, Callable

import numpy as np
from oricreate.crease_pattern import \
    CreasePatternState
from oricreate.forming_tasks import \
    FactoryTask
import sympy as sp


x_, y_ = sp.symbols('x, y')


class HexagonalCPFactory(FactoryTask):

    '''Generate a Yoshimura crease pattern based
    on the specification of its parameters.

    .. todo::
       Default parameters do not work. d_x not described.
    '''

    L_x = Float(4, geometry=True)
    L_y = Float(4, geometry=True)

    n_seg = Int(4)

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

    L_rigid = Property

    def _get_L_rigid(self):
        return self._geometry[3]

    N_x_sym = Property

    def _get_N_x_sym(self):
        return self._geometry[4]

    N_up = Property

    def _get_N_up(self):
        return self._geometry[5]

    N_down = Property

    def _get_N_down(self):
        return self._geometry[6]

    geo_transform = Callable

    def _geo_transform_default(self):
        return lambda X_arr: X_arr

    _geometry = Property(depends_on='+geometry')

    @cached_property
    def _get__geometry(self):

        n_seg = self.n_seg
        n_x = n_seg * n_seg + 1
        n2 = n_x / 2

        L_x = 1.0
        L_y = 1.0

        # provide the base element with four by four discretization

        x_e, y_e = np.mgrid[0:L_x:complex(n_x), 0:L_y:complex(n_x)]

        x_m = (x_e[:-1, :-1] + x_e[1:, 1:]) / 2.0
        y_m = (y_e[:-1, :-1] + y_e[1:, 1:]) / 2.0

        x1 = np.c_[x_e.flatten(), y_e.flatten(),
                   np.zeros_like(x_e.flatten())]
        x2 = np.c_[x_m.flatten(), y_m.flatten(),
                   np.zeros_like(x_m.flatten())]
        x = np.vstack([x1, x2])

        Nf1 = np.arange(n_x * n_x).reshape(n_x, n_x)
        n_x2 = n_x - 1

        def get_facets(N1):
            f1 = np.array(
                [N1[:-1, :-1].flatten(),
                 N1[1:, :-1].flatten(),
                 N1[1:, 1:].flatten()]).T

            f2 = np.array(
                [N1[:-1, :-1].flatten(),
                 N1[1:, 1:].flatten(),
                 N1[:-1, 1:].flatten()]).T

            return np.vstack([f1, f2])

        ff1 = get_facets(Nf1[:n2 + 1, :n2 + 1])
        ff2 = get_facets(Nf1[n2:, n2:])
        nlh = Nf1[:, n2]
        nlv = Nf1[n2, ::-1]
        f5 = np.array(
            [nlh[:n2].flatten(),
             nlv[:n2].flatten(),
             nlh[1:n2 + 1].flatten()]).T
        f6 = np.array(
            [nlh[n2 + 1:-1].flatten(),
             nlv[n2 + 1:-1].flatten(),
             nlh[n2 + 2:].flatten()]).T
        f7 = np.array(
            [nlv[:n2 - 1].flatten(),
             nlh[1:n2].flatten(),
             nlv[1:n2].flatten()]).T
        f8 = np.array(
            [nlv[n2:- 1].flatten(),
             nlh[n2 + 1:].flatten(),
             nlv[n2 + 1:].flatten()]).T

        nl_fixed = np.vstack([f5[:-1, (1, 2)], f6[:, (1, 2)]])

        facets = np.vstack([ff1, ff2, f5, f6, f7, f8])

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

        l_mapping = np.array(
            np.bincount(l_arr, np.arange(len(l_arr))),
            dtype=np.int_)
        l_fixed = NN_L[nl_fixed[:, 0], nl_fixed[:, 1]]
        l_fixed_red = l_mapping[l_fixed]
        # shrink / condense
        N_connected = np.where(np.sum(NN_L + 1, axis=1) > 0)[0]
        N_enum = np.zeros(n_N, dtype=np.int_) - 1
        N_enum[N_connected] = np.arange(len(N_connected))

        Nf1_x_sym = Nf1[np.arange(len(Nf1)), np.arange(len(Nf1))]
        Nf_x_sym = N_enum[np.hstack([Nf1_x_sym])]

        x_red = x[N_connected, :]
        l_red = N_enum[lines]
        f_red = N_enum[facets]

        s = Nf1.shape
        i, j = np.mgrid[:(s[0] + 1) / 2, :(s[1] + 1) / 2]
        i_arr, j_arr = np.where(i[:, ::2] >= j[:, ::2])
        Nuph1 = N_enum[Nf1[i_arr, j_arr * 2]]
        Nuph2 = N_enum[Nf1[-i_arr - 1, -j_arr * 2 - 1]]
        Nupv1 = N_enum[Nf1[j_arr * 2, i_arr]]
        Nupv2 = N_enum[Nf1[-j_arr * 2 - 1, -i_arr - 1]]

        print('N_uph1', Nuph1)

        Nf_up = np.unique(np.hstack([Nuph1, Nuph2, Nupv1, Nupv2]))

        i, j = np.mgrid[:(s[0]) / 2, :(s[1]) / 2]
        i_arr, j_arr = np.where(i[:, ::2] >= j[:, ::2])
        Ndoh1 = N_enum[Nf1[i_arr + 1, (j_arr * 2) + 1]]
        Ndoh2 = N_enum[Nf1[-i_arr - 2, -j_arr * 2 - 2]]
        Ndov1 = N_enum[Nf1[j_arr * 2 + 1, i_arr + 1]]
        Ndov2 = N_enum[Nf1[-j_arr * 2 - 2, -i_arr - 2]]

        Nf_do = np.unique(np.hstack([Ndoh1, Ndoh2, Ndov1, Ndov2]))

        x_red = self.geo_transform(x_red)

        return (x_red, l_red, f_red, l_fixed_red,
                Nf_x_sym, Nf_up, Nf_do)


if __name__ == '__main__':

    def geo_transform(x_arr):
        alpha = np.pi / 4.0
        L_x = 6.0
        L_y = 2.0
        x_max = np.max(x_arr, axis=0)
        x_min = np.min(x_arr, axis=0)
        T = (x_max - x_min) / 2.0
        x_arr -= T[np.newaxis, :]

        R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]], dtype=np.float_)
        x_rot = np.einsum('ij,nj->ni', R, x_arr)
        x_rot[:, 0] *= L_x
        x_rot[:, 1] *= L_y
        return x_rot

    yf = HexagonalCPFactory(L_x=2,
                            L_y=1,
                            n_seg=2,
                            geo_transform=geo_transform
                            )

    cp = yf.formed_object
    print(yf.L_rigid)
    print('N_x_sym', yf.N_x_sym)
    print(yf.N_up)
    print(yf.N_down)

    import pylab as p
    cp.plot_mpl(p.axes(), nodes=True, lines=True, facets=False)
    p.show()
