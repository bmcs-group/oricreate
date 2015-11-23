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
'''
Created on 09.12.2014

@author: gilbe_000
'''

from traits.api import \
    Float, Int, Property, cached_property, Callable

import numpy as np
from oricreate.crease_pattern import \
    CreasePatternState
from oricreate.forming_tasks import \
    FactoryTask
import sympy as sp


x_, y_ = sp.symbols('x, y')


class RonReshCPFactory(FactoryTask):

    '''Generate a Yoshimura crease pattern based
    on the specification of its parameters.

    .. todo::
       redefine the factory to provide a hexagonal as an atomic unit.
    '''

    L_x = Float(4, geometry=True)

    n_x = Int(2, geometry=True)
    n_y = Int(2, geometry=True)

    def deliver(self):
        return CreasePatternState(X=self.X,
                                  L=self.L,
                                  F=self.F)

    X = Property

    def _get_X(self):
        return self._geometry[0]

    def _set_X(self, values):
        values = values.reshape(-1, 3)
        self.X[:, :] = values[:, :]

    L = Property

    def _get_L(self):
        return self._geometry[1]

    F = Property

    def _get_F(self):
        return self._geometry[2]

    N_h = Property

    def _get_N_h(self):
        return self._geometry[3]

    N_v = Property

    def _get_N_v(self):
        return self._geometry[4]

    N_i = Property

    def _get_N_i(self):
        return self._geometry[5]

    X_h = Property

    def _get_X_h(self):
        return self._geometry[6]

    X_v = Property

    def _get_X_v(self):
        return self._geometry[7]

    X_i = Property

    def _get_X_i(self):
        return self._geometry[8]

    interior_vertices = Property

    def _get_interior_vertices(self):
        return self._geometry[9]

    cycled_neighbors = Property

    def _get_cycled_neighbors(self):
        return self._geometry[10]

    geo_transform = Callable

    def _geo_transform_default(self):
        return lambda X_arr: X_arr

    _geometry = Property(depends_on='+geometry')

    @cached_property
    def _get__geometry(self):

        n_x = self.n_x
        n_y = self.n_y

        L_x = self.L_x
        L_y = np.sqrt(3) * L_x * 6 / 9 / n_x * n_y

        # base grid
        x_1, y_1 = np.mgrid[
            0:L_x:complex(n_x * 4 + 1), 0:L_y:complex(n_y * 4 + 1)]

        # Changes

        y_1[0::4, 0::2] = y_1[0::4, 0::2] + np.sqrt(3) * L_x / 18 / n_x
        y_1[2::4, 0::2] = y_1[2::4, 0::2] - np.sqrt(3) * L_x / 9 / n_x
        y_1[2::4, 1::2] = y_1[2::4, 1::2] - np.sqrt(3) * L_x / 6 / n_x
        x_1[1::4, 0::2] = x_1[1::4, 0::2] - (L_x * 1 / 12 / n_x)
        x_1[1::4, 1::2] = x_1[1::4, 1::2] + (L_x * 1 / 12 / n_x)
        x_1[3::4, 1::2] = x_1[3::4, 1::2] - (L_x * 1 / 12 / n_x)
        x_1[3::4, 0::2] = x_1[3::4, 0::2] + (L_x * 1 / 12 / n_x)

        X_all = np.c_[x_1.flatten(), y_1.flatten()]

        # nodes on vertical boundaries on odd horizontal crease lines

        # X_v_1 = X_1[(0, 1, -2, -1), 0::1]
        # X_v_2 = X_3[(0, 1, 2, -3, -2, -1), 0::1]

        # X_v = np.vstack((X_v_1, X_v_2))

        # interior nodes on odd horizontal crease lines

        # x_i = (x_e[1:, 1::2] + x_e[:-1, 1::2]) / 2.0
        # y_i = (y_e[1:, 1::2] + y_e[:-1, 1::2]) / 2.0
        # X_i = np.c_[x_i.flatten(), y_i.flatten()]

        # n_all = np.arange((n_x + 1) * (n_y + 1)).reshape((n_x + 1), (n_y + 1))

        nodes = X_all

        zero_z = np.zeros((nodes.shape[0], 1), dtype=float)

        nodes = np.hstack([nodes, zero_z])

        n_all = np.arange(
            (n_x * 4 + 1) * (n_y * 4 + 1)).reshape((n_x * 4 + 1),
                                                   (n_y * 4 + 1))

        # connectivity of nodes defining the crease pattern

        # ======================================================================
        # Construct the creaseline mappings
        # ======================================================================

        c_h = np.column_stack((np.arange(
            (n_x * 4 + 1) * (n_y * 4 + 1) - (n_y * 4 + 1)),
            np.arange(n_y * 4 + 1, (n_x * 4 + 1) * (n_y * 4 + 1))))

        c_v = np.column_stack(
            (n_all[:, 0:-1].flatten('F'), n_all[:, 1:].flatten('F')))

        c_d_1 = np.column_stack(
            (n_all[0:-1:4, 0:-1:1].flatten(), n_all[1::4, 1::1].flatten()))
        c_d_2 = np.column_stack(
            (n_all[1::4, 0:-1:1].flatten(), n_all[2::4, 1::1].flatten()))
        c_d_3 = np.column_stack(
            (n_all[3::4, 1::1].flatten(), n_all[4::4, 0:-1:1].flatten()))
        c_d_4 = np.column_stack(
            (n_all[2::4, 1::1].flatten(), n_all[3::4, 0:-1:1].flatten()))

        crease_lines = np.vstack((c_h, c_v, c_d_1, c_d_2, c_d_3, c_d_4))

        # ======================================================================
        # Construct the facet mappings
        # ======================================================================
        # print "facetten", n_all[1::4, 1::1].flatten()
        facets_1 = np.column_stack(
            (n_all[0:-1:4, 0:-1:1].flatten(), n_all[0:-1:4, 1::1].flatten(),
             n_all[1::4, 1::1].flatten()))
        facets_2 = np.column_stack(
            (n_all[0:-1:4, 0:-1:1].flatten(), n_all[1::4, 1::1].flatten(),
             n_all[1::4, 0:-1:1].flatten()))
        facets_3 = np.column_stack(
            (n_all[1::4, 0:-1:1].flatten(), n_all[1::4, 1::1].flatten(),
             n_all[2::4, 1::1].flatten()))
        facets_4 = np.column_stack(
            (n_all[1::4, 0:-1:1].flatten(), n_all[2::4, 1::1].flatten(),
             n_all[2::4, 0:-1:1].flatten()))
        facets_5 = np.column_stack(
            (n_all[2::4, 0:-1:1].flatten(), n_all[2::4, 1::1].flatten(),
             n_all[3::4, 0:-1:1].flatten()))
        facets_6 = np.column_stack(
            (n_all[2::4, 1::1].flatten(), n_all[3::4, 1::1].flatten(),
             n_all[3::4, 0:-1:1].flatten()))
        facets_7 = np.column_stack(
            (n_all[3::4, 0:-1:1].flatten(), n_all[3::4, 1::1].flatten(),
             n_all[4::4, 0:-1:1].flatten()))
        facets_8 = np.column_stack(
            (n_all[3::4, 1::1].flatten(), n_all[4::4, 1::1].flatten(),
             n_all[4::4, 0:-1:1].flatten()))

        facets = np.vstack(
            (facets_1, facets_2, facets_3, facets_4,
             facets_5, facets_6, facets_7, facets_8))

        return (nodes, crease_lines, facets,
                )
#         return (self.geo_transform(nodes), crease_lines, facets,
#                 n_h, n_v, n_i, X_h, X_v, X_i,
#                 interior_vertices, cycled_neighbors)

if __name__ == '__main__':

    yf = RonReshCPFactory(n_x=1,
                          n_y=1,
                          L_x=1,
                          )

    cp = yf.formed_object

    import pylab as p
    cp.plot_mpl(p.axes())
    p.show()
