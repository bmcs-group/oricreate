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


class FuPotEngNodeLoadViz3D(Viz3D):
    '''Visualize the crease Pattern
    '''

    def get_node_load_values(self, t):
        fu_tot_poteng = self.vis3d
        ft = fu_tot_poteng.forming_task
        cp = ft.formed_object

        n = np.array([node for node, dim, value
                      in fu_tot_poteng.F_ext_list], dtype='int_')
        dims = np.array([dim for node, dim, value
                         in fu_tot_poteng.F_ext_list], dtype='int_')
        values = np.array([value(t) for node, dim, value
                           in fu_tot_poteng.F_ext_list], dtype='float_')
        x_n = cp.x[n, :]
        x0_n = np.copy(x_n)
        x0_n[np.arange(len(values)), dims] -= values
        u_n = x_n - x0_n
        x, y, z = x_n.T
        u, v, w = u_n.T

        return x, y, z, u, v, w

    def plot(self):

        m = self.ftv.mlab
        x, y, z, u, v, w = self.get_node_load_values(1.0)

        cl_arrow = m.quiver3d(x, y, z, u, v, w, mode='arrow',
                              color=(1.0, 0.0, 0.0),
                              scale_mode='vector')
        cl_arrow.glyph.glyph_source.glyph_position = 'head'
        self.pipes['cl_arrow'] = cl_arrow
        self.pipes['surf'] = m.pipeline.surface(cl_arrow)

    def update(self, vot=0.0):
        x, y, z, u, v, w = self.get_node_load_values(1.0)
        cl_arrow = self.pipes['cl_arrow']
        cl_arrow.mlab_source.set(x=x, y=y, z=z, u=u, v=v, w=w)
