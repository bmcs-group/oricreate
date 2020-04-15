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
# Created on Jan 3, 2013 by:  schmerl

from copy import copy
import math
import os
import tempfile
from traits.api import Property, cached_property, \
    Array, Int, \
    Str, Float, Dict, WeakRef

from .exporter import Exporter
import numpy as np
from oricreate.viz3d import Visual3D


class MeshExporter(Exporter, Visual3D):
    '''Generate a triangular mesh.
    '''
    n_l_e = Int(2, label='Numbere of elements along a crease line',
                geometry=True)

    def _get_geometry(self):
        ft = self.forming_task
        cp = ft.formed_object
        eta_range = np.linspace(0, 1, self.n_l_e + 1)
        L_N_X = cp.x[cp.L_N]

        # subdivide all lines
        # @todo - include only lines connected to a facet.
        # @todo - define x_1 as a final configuration of a forming_task
        L_K_X0 = np.einsum('k,ld->lkd', (1 - eta_range), L_N_X[:, 0, :])
        L_K_X1 = np.einsum('k,ld->lkd', eta_range, L_N_X[:, 1, :])
        L_K_X = L_K_X0 + L_K_X1

        # number of inner nodes on all lines
        n_L_iN = cp.n_L * (self.n_l_e - 1)
        # array mapping the line number and node with a line to global node
        L_n_N = np.zeros((cp.n_L, self.n_l_e + 1))
        L_n_N[:, (0, -1)] = cp.L_N
        L_n_N[:, 1:-1] = np.arange(n_L_iN).reshape(cp.n_L, -1) + cp.n_N
        N_plus = L_n_N[:, :]
        N_minus = L_n_N[:, ::-1]
        NN_L_n_N = np.zeros((cp.n_N, cp.n_N, self.n_l_e + 1))

        NN_L_n_N[cp.L[:, 0], cp.L[:, 1]] = N_plus
        NN_L_n_N[cp.L[:, 1], cp.L[:, 0]] = N_minus

        # subdivide all facets
        eta2, eta3 = np.mgrid[0:1:np.complex(self.n_l_e + 1),
                              0:1:np.complex(self.n_l_e + 1)]
        eta1 = 1 - eta2 - eta3
        inner = np.where((eta1 > 1e-5) & (eta2 > 1e-5) & (eta3 > 1e-5))

        eta = np.vstack([eta1[inner],
                         eta2[inner],
                         eta3[inner]]).T

        F_N_x = cp.x[cp.F_N]
        F_r = np.einsum('an,fnd->fad', eta, F_N_x)

        N_X = np.vstack([cp.x,
                         L_K_X[:, 1:-1, :].reshape(-1, cp.n_D),
                         F_r.reshape(-1, cp.n_D)],
                        )

        # Enumerate facet nodes - inherit node and line nodes
        # stored in NN_L_n_N
        n_n_e = self.n_l_e + 1
        F_n_N = np.zeros((cp.n_F, n_n_e, n_n_e), dtype=np.int_) - 1

        didx = np.arange(n_n_e)
        F_n_N[:, didx, 0] = NN_L_n_N[cp.F_L_N[:, 0, 0], cp.F_L_N[:, 0, 1]]
        F_n_N[
            :, -didx - 1, didx] = NN_L_n_N[cp.F_L_N[:, 1, 0], cp.F_L_N[:, 1, 1]]
        F_n_N[:, 0, -didx - 1] = NN_L_n_N[cp.F_L_N[:, 2, 0], cp.F_L_N[:, 2, 1]]

        n_F_iN = cp.n_F * len(inner[0])
        F_n_N[(slice(None),) + inner] = \
            np.arange(n_F_iN).reshape(cp.n_F, -1) + cp.n_N + n_L_iN

        # Triangulate
        #
        F_t1_N = np.array([F_n_N[:, :-1, :-1],
                           F_n_N[:, 1:, :-1],
                           F_n_N[:, :-1, 1:]]).transpose([1, 2, 3, 0])
        F_t2_N = np.array([F_n_N[:, 1:, :-1],
                           F_n_N[:, 1:, 1:],
                           F_n_N[:, :-1, 1:]]).transpose([1, 2, 3, 0])

        eta3, eta2 = np.mgrid[0:1:np.complex(self.n_l_e),
                              0:1:np.complex(self.n_l_e)]
        eta1 = 1 - eta2 - eta3
        f1, f2 = np.where((eta1 >= 0) & (eta2 >= 0) & (eta3 >= 0))
        F1 = F_t1_N[:, f1, f2, :]

        eta3, eta2 = np.mgrid[0:1:np.complex(self.n_l_e - 1),
                              0:1:np.complex(self.n_l_e - 1)]
        eta1 = 1 - eta2 - eta3
        f1, f2 = np.where((eta1 >= 0) & (eta2 >= 0) & (eta3 >= 0))
        F2 = F_t2_N[:, f1, f2, :]

        F_t_N = np.hstack([F1, F2]).reshape(-1, 3)

        return N_X, F_t_N

    def plot_mlab(self, m, nodes=False):
        m.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

        X, F = self._get_geometry()
        x, y, z = X.T

        m.triangular_mesh(x, y, z, F,
                          line_width=3,
                          representation='wireframe',
                          color=(0.6, 0.6, 0.6))

        if nodes:
            text = np.array([])
            pts = X
            for i in range(len(pts)):
                temp_text = m.text3d(
                    x[i], y[i], z[i] + 0.2, str(i), scale=0.05)


class InfoCadMeshExporter(MeshExporter):

    def write(self):

        X, F = self._get_geometry()
        x, y, z = X.T
        nodes = "*Node"
        for i in range(len(x)):
            temp_node = ' %i \t %.4f \t %.4f \t %.4f\n' % (
                i + 1, x[i], y[i], z[i])
            temp_node = temp_node.replace('.', ',')
            nodes += temp_node

        F0, F1, F2 = F.T + 1
        elements = "*Elements"
        for i in range(len(F0)):
            temp_facet = ' %i\tSH36\t%i\t%i\t%i\t\t\t\t\t\t1\n' % \
                (i + 1, F0[i], F1[i], F2[i])
            elements += temp_facet

        part = '*Part, NAME=Part-1\n'
        part += nodes
        part += elements

        fname_base = 'infocad_mesh'
        tdir = tempfile.mkdtemp()
        fname_path = os.path.join(tdir, fname_base)

        fname = fname_path + '.inp'
        inp_file = open(fname, 'w')
        inp_file.write(part)
        inp_file.close()
        print('inp file %s written' % fname)

if __name__ == '__main__':
    from oricreate.api import YoshimuraCPFactory, \
        CreasePattern, CustomCPFactory
    cpf = YoshimuraCPFactory(L_x=4, L_y=2, n_x=3, n_y=4)
#     cp = CreasePattern(X=[[0, 0, 0],
#                           [2, 0, 0],
#                           [0, 2, 0],
#                           ],
#                        L=[[0, 1], [2, 0], [2, 1]],
#                        F=[[0, 1, 2]])
#     cpf = CustomCPFactory(formed_object=cp)
    cpf.formed_object.x[20, 2] = 0.5
    me = InfoCadMeshExporter(forming_task=cpf, n_l_e=8)

    me.write()

    X, F = me._get_geometry()

    x, y, z = X.T

    import mayavi.mlab as m
    me.plot_mlab(m)
    m.show()
