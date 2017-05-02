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
from traits.api import Array

from exporter import Exporter
import numpy as np
from oricreate.viz3d import Visual3D


class ScaffoldingExporter(Exporter, Visual3D):
    '''Generate a triangular mesh.
    '''

    def generate_scaffolding(self, x_scaff_position):

        ft = self.forming_task
        cp = ft.formed_object
        x_1 = ft.x_1
        L = cp.L

        p_0 = np.array([x_scaff_position, 0, 0], dtype='float_')
        l_0 = x_1[L[:, 0]]
        l = cp.L_vectors
        n = np.array([1.0, 0, 0], dtype='float_')

        p_0_l_0 = p_0[np.newaxis, :] - l_0
        nom = np.einsum('...i,...i->...', p_0_l_0, n[np.newaxis, :])
        denom = np.einsum('...i,...i->...', l, n[np.newaxis, :])
        d = nom / denom

        l_idx = np.where((d < 1.001) & (d > -0.001))[0]
        p = d[l_idx, np.newaxis] * l[l_idx, :] + l_0[l_idx, :]

        x_values, y_values = p[:, (1, 2)].T
        six = np.argsort(x_values)
        sx_values, sy_values = x_values[six], y_values[six]
        spoints = np.c_[sx_values, sy_values]
        left_points, right_points = spoints[:-1, :], spoints[1:, :]
        svects = right_points - left_points
        norm_svects = np.linalg.norm(svects, axis=1)
        l_idx = np.where(norm_svects > 0.01)[0]

        last_x = sx_values[l_idx[-1] + 1]
        last_y = sy_values[l_idx[-1] + 1]
        x, y = sx_values[l_idx], sy_values[l_idx]

        return np.hstack([x, [last_x]]), np.hstack([y, [last_y]])

    scaff_positions = Array

    def _scaff_positions_default(self):

        pos = np.array(
            [-1.065, -0.77, -0.45, -0.25, 0.25, 0.45, 0.77, 1.065], dtype='float_')
        return pos

    scaff_ref_nodes = Array

    def _scaff_ref_nodes_default(self):
        return np.array([14, 41, 18, 45, 21], dtype='int_')

    def generate_scaffoldings(self):

        L_mid = 1.5

        ft = self.forming_task
        x_1 = ft.x_1
        x_42 = x_1[42][0]

        ref_lines = np.c_[self.scaff_ref_nodes[:-1], self.scaff_ref_nodes[1:]]
        # print 'ref', ref_lines
        ref_midpoints = (x_1[ref_lines[:, 1]] + x_1[ref_lines[:, 0]]) / 2.0
        # print 'mp', ref_midpoints[:, 0]

        scaff_positions = ref_midpoints[:, 0] - L_mid
        # print 'sp', scaff_positions

        #scaff_positions = self.scaff_positions
        scaff_positions = np.hstack([[0, x_42 - L_mid], scaff_positions])

        #centered_pos = pos + offset

        centered_pos = np.hstack([[L_mid, x_42], ref_midpoints[:, 0]])
        print 'cp', centered_pos

        scaff_plates = []
        min_max = []
        for s_pos in centered_pos:
            x, y = self.generate_scaffolding(s_pos)
            min_max.append([np.min(x), np.max(x), np.min(y), np.max(y)])
            scaff_plates.append(np.array([x, y]))

        min_max_arr = np.array(min_max)
        min_x = np.min(min_max_arr[:, 0])
        max_x = np.max(min_max_arr[:, 1])
        min_y = np.min(min_max_arr[:, 2])
#        max_y = np.max(min_max_arr[:, 3])
        min_scaff_height = 0.05

        x_0 = (min_x + max_x) / 2.0
        y_0 = min_y + min_scaff_height

        scaff_plates_0 = [[x - x_0, y - y_0] for x, y in scaff_plates]

        import tempfile
        import os.path
        import pylab as p

        tdir = tempfile.mkdtemp()
        for idx, (x_y, s_pos) in enumerate(zip(scaff_plates_0, scaff_positions)):
            x, y = x_y
            x_close = [np.max(x), np.min(x)]
            y_close = [0, 0]

            p.clf()
            x_p = np.hstack([x, x_close, [x[0]]])
            y_p = np.hstack([y, y_close, [y[0]]])

            ax = p.axes()
            ax.axis('equal')

            ax.plot(x_p, y_p)
            for x_v, y_v in zip(x, y):
                ax.annotate('%5.3f,%5.3f' %
                            (x_v, y_v), xy=(x_v, y_v), rotation=90)
            ax.annotate('scaffold x - position %5.3f' % -s_pos, xy=(0, 0.04))

            fname_path = os.path.join(tdir, 'scaff%d.pdf' % idx)
            print 'saving in %s', fname_path
            p.savefig(fname_path)

        p.show()

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
