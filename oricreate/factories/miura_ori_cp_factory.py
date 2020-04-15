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


class MiuraOriCPFactory(FactoryTask):

    '''Generate a Yoshimura crease pattern based
    on the specification of its parameters.

    .. todo::
       Default parameters do not work. d_x not described.
    '''

    L_x = Float(4, geometry=True)
    L_y = Float(2, geometry=True)

    n_x = Int(2, geometry=True)
    n_y = Int(2, geometry=True)
    d_0 = Float(-0.1, geometry=True)
    d_1 = Float(0.1, geometry=True)

    def deliver(self):
        return CreasePatternState(X=self.X,
                                  L=self.L,
                                  F=self.F)

    X = Property

    def _get_X(self):
        return self._geometry[0]

    def _set_X(self, values):
        values = values.reshape(-1, 3)
        self.X[...] = values[...]

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

    N_grid = Property(depends_on='+geometry')
    '''Node numbers ordered in a two-dimensional grid.
    '''
    @cached_property
    def _get_N_grid(self):
        n_N = (self.n_x + 1) * (self.n_y + 1)
        return np.arange(n_N).reshape(self.n_x + 1, self.n_y + 1)

    L_N_v = Property(depends_on='+geometry')
    '''Vertical lines-node connectivity.
    '''
    @cached_property
    def _get_L_N_v(self):
        N_grid = self.N_grid
        return np.c_[N_grid[:, :-1].flatten(), N_grid[:, 1:].flatten()]

    L_v_grid = Property(depends_on='+geometry')
    '''Vertical line indices ordered in a two-dimensional grid..
    '''
    @cached_property
    def _get_L_v_grid(self):
        return np.arange(len(self.L_N_v)).reshape(self.n_x + 1, self.n_y)

    L_N_h = Property(depends_on='+geometry')
    '''Horizontal lines - node connectivity..
    '''
    @cached_property
    def _get_L_N_h(self):
        N_grid = self.N_grid
        return np.c_[N_grid[:-1, :].flatten(), N_grid[1:, :].flatten()]

    L_h_grid = Property(depends_on='+geometry')
    '''Horizontal line indices ordered in a two-dimensional grid..
    '''
    @cached_property
    def _get_L_h_grid(self):
        offset = len(self.L_N_v)
        L_h = np.arange(len(self.L_N_h)) + offset
        return L_h.reshape(self.n_x, self.n_y + 1)

    L_N_d = Property(depends_on='+geometry')
    '''Diagonal lines-node connectivity..
    '''
    @cached_property
    def _get_L_N_d(self):
        N_grid = self.N_grid
        return np.c_[N_grid[1:, :-1].flatten(), N_grid[:-1, 1:].flatten()]

    L_d_grid = Property(depends_on='+geometry')
    '''Diagonal line indices ordered in a two-dimensional grid..
    '''
    @cached_property
    def _get_L_d_grid(self):
        offset = len(self.L_N_v) + len(self.L_N_h)
        L_d = np.arange(len(self.L_N_d)) + offset
        return L_d.reshape(self.n_x, self.n_y)

    _geometry = Property(depends_on='+geometry')

    @cached_property
    def _get__geometry(self):

        n_x = self.n_x
        n_y = self.n_y

        L_x = self.L_x
        L_y = self.L_y

        x_e, y_e = np.mgrid[0:L_x:complex(n_x + 1), 0:L_y:complex(n_y + 1)]
        z_e = np.zeros_like(x_e)

        x_e[1:-1, ::2] += self.d_0
        x_e[1:-1, 1::2] += self.d_1

        X_h = np.c_[x_e.flatten(), y_e.flatten(), z_e.flatten()]

        nodes = X_h

        # connectivity of nodes defining the crease pattern

        # ======================================================================
        # Construct the creaseline mappings
        # ======================================================================

        nidx = np.arange(len(nodes)).reshape(n_x + 1, n_y + 1)

        l_vertical = np.c_[nidx[:, :-1].flatten(), nidx[:, 1:].flatten()]
        l_horizontal = np.c_[nidx[:-1, :].flatten(), nidx[1:, :].flatten()]
        l_diag = np.c_[nidx[1:, :-1].flatten(), nidx[:-1, 1:].flatten()]

        lines = np.vstack([l_vertical, l_horizontal, l_diag])

        # ======================================================================
        # Construct the facet mappings
        # ======================================================================
        f_1 = np.c_[
            nidx[:-1, :-1].flatten(), nidx[1:, :-1].flatten(), nidx[:-1, 1:].flatten()]
        f_2 = np.c_[
            nidx[1:, :-1].flatten(), nidx[1:, 1:].flatten(), nidx[:-1, 1:].flatten()]

        facets = np.vstack((f_1, f_2))

        return (nodes, lines, facets,
                )

if __name__ == '__main__':

    yf = MiuraOriCPFactory(L_x=30,
                           L_y=15,
                           n_x=3,
                           n_y=3,
                           d_0=1.0,
                           d_1=-1.0)

    print(yf.N_grid[:, 0])
    print(yf.L_d_grid[:, 0])
    cp = yf.formed_object

    print(cp.F)

    import pylab as p
    cp.plot_mpl(p.axes())
    p.show()
