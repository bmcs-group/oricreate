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
# Created on Sep 7, 2011 by: rch

from traits.api import \
    Float, Int, Property, cached_property, Callable
import numpy as np
import sympy as sp

from oricreate.crease_pattern import CreasePattern


from oricreate.forming_tasks import \
    FactoryTask

x_, y_ = sp.symbols('x, y')


class MiuraOriCPFactory(FactoryTask):

    '''Generate a Yoshimura crease pattern based
    on the specification of its parameters.
    '''

    L_x = Float(4, geometry=True)
    L_y = Float(2, geometry=True)

    n_x = Int(2, geometry=True)
    n_y = Int(2, geometry=True)
    d_x = Float(4)

    def _get_formed_object(self):
        return CreasePattern(X=self.X,
                             L=self.L,
                             F=self.F)

    X = Property

    def _get_X(self):
        return self._geometry[0]

    def _set_X(self, values):
        values = values.reshape(-1, 3)
        self.X[:, :] = values[:,:]

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
        d_x = self.d_x

        L_x = self.L_x
        L_y = self.L_y

        x_e, y_e = np.mgrid[0:L_x:complex(n_x + 1), 0:L_y:complex(n_y + 1)]
        # print x_e
        # print y_e
        x_e[1:-1:1, 0::2] = x_e[1:-1:1, 0::2] + d_x
        x_e[1:-1:1, 1::2] = x_e[1:-1:1, 1::2] - d_x

        # nodes on horizontal crease lines

        x_h = x_e[:, ::1]

        y_h = y_e[:, ::1]
        X_h = np.c_[x_h.flatten(), y_h.flatten()]

        # nodes on vertical boundaries on odd horizontal crease lines

        x_v = x_e[(0, -1), 1::1]
        y_v = y_e[(0, -1), 1::1]
        X_v = np.c_[x_v.flatten(), y_v.flatten()]

        # interior nodes on odd horizontal crease lines

        x_i = (x_e[1:, 1::2] + x_e[:-1, 1::2]) / 2.0
        y_i = (y_e[1:, 1::2] + y_e[:-1, 1::2]) / 2.0
        X_i = np.c_[x_i.flatten(), y_i.flatten()]

        n_all = np.arange((n_x + 1) * (n_y + 1)).reshape((n_x + 1), (n_y + 1))

        nodes = X_h

        zero_z = np.zeros((nodes.shape[0], 1), dtype=float)

        nodes = np.hstack([nodes, zero_z])

        # connectivity of nodes defining the crease pattern

        # ======================================================================
        # Construct the creaseline mappings
        # ======================================================================

        c_h = np.column_stack((np.arange(
            (n_x + 1) * (n_y + 1) - n_y - 1), np.arange(n_y + 1, (n_x + 1) * (n_y + 1))))
        c_v = np.column_stack(
            (n_all[:, 0:-1].flatten('F'), n_all[:, 1:].flatten('F')))
        c_d = np.column_stack(
            (n_all[0:-1:1, 0:-1:1].flatten(), n_all[1::1, 1::1].flatten()))

        crease_lines = np.vstack((c_h, c_v, c_d))

        # ======================================================================
        # Construct the facet mappings
        # ======================================================================
        f_1 = n_all[0:-1:1, 0:-1:1].flatten()
        f_2 = n_all[0:-1:1, 1::1].flatten()
        f_3 = n_all[1::1, 1::1].flatten()
        f_4 = n_all[1::1, 0:-1:1].flatten()

        facets_1 = np.column_stack((f_1, f_2, f_3))
        facets_2 = np.column_stack((f_3, f_4, f_1))
        facets = np.vstack((facets_1, facets_2))
        print "facetten", facets

        return (nodes, crease_lines, facets,
                )
#         return (self.geo_transform(nodes), crease_lines, facets,
#                 n_h, n_v, n_i, X_h, X_v, X_i,
#                 interior_vertices, cycled_neighbors)

if __name__ == '__main__':

    yf = MiuraOriCPFactory(L_x=3,
                           L_y=3,
                           n_x=3,
                           n_y=4,
                           d_x=0.15)

    cp = yf.formed_object

    import pylab as p
    cp.plot_mpl(p.axes())
    p.show()
