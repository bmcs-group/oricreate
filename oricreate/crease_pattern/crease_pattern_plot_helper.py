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
    Float, Property, cached_property

from .crease_pattern_viz3d import \
    CreasePatternViz3D, CreasePatternNodeNumbersViz3D, \
    CreasePatternDisplViz3D
import numpy as np
from oricreate.viz3d import \
    Visual3D


class CreasePatternPlotHelper(Visual3D):

    '''
    Methods exporting the crease pattern geometry into other formats.
    '''

    x_t = Property

    def _get_x_t(self):
        return np.array([self.x], dtype='float_')

    def plot_mpl(self, ax, nodes=True, lines=True, facets=True,
                 color='black', linewidth=1, fontsize=12):
        r'''Plot the crease pattern using mpl
        '''
        # set plot range
        x_max = np.amax(self.x_0[:, :2], axis=0)
        x_min = np.amin(self.x_0[:, :2], axis=0)
        x_delta = x_max - x_min
        larger_range = np.amax(x_delta)
        pad = larger_range * 0.1
        x_mid = (x_max + x_min) / 2.0

        shift = np.array([-0.5, 0.5], dtype='f')
        padding = np.array([-pad, pad], dtype='f')
        x_deltas = (x_delta[:, np.newaxis] * shift[np.newaxis, :] +
                    padding[np.newaxis, :])
        x_range = x_mid[:, np.newaxis] + x_deltas

        # plot lines
        crease_line_pos = self.x_0[:, (0, 1)][self.cL_N]
        ax.plot(crease_line_pos[:, :, 0].T,
                crease_line_pos[:, :, 1].T, color=color,
                linewidth=linewidth)
        ax.set_xlim(*x_range[0, :])
        ax.set_ylim(*x_range[1, :])
        ax.set_aspect('equal')
        # plot node numbers
        if nodes is True:
            xy_offset = (3, 3)
            for n, x_0 in enumerate(self.x_0):
                xy = (x_0[0], x_0[1])
                ax.annotate(xy=xy, s='%g' % n,
                            xytext=xy_offset, color='blue',
                            textcoords='offset points', size=fontsize)
        # plot line numbers
        if lines is True:
            xy_offset = (1, 1)
            crease_line_pos = 0.5 * np.sum(self.x_0[self.cL_N], axis=1)
            for n, x_0 in zip(self.cL, crease_line_pos):
                xy = (x_0[0], x_0[1])
                ax.annotate(xy=xy, s='%g' % n,
                            xytext=xy_offset, color='red',
                            textcoords='offset points', size=fontsize)
        # plot facet numbers
        if facets is True:
            xy_offset = (0, 0)
            facet_pos = 1 / 3.0 * np.sum(self.x_0[self.F], axis=1)
            for n, x_0 in enumerate(facet_pos):
                xy = (x_0[0], x_0[1])
                ax.annotate(xy=xy, s='%g' % n,
                            xytext=xy_offset, color='green',
                            textcoords='offset points', size=fontsize)

    def _get_bounding_box(self):
        return np.min(self.x, axis=0), np.max(self.x, axis=0)

    def _get_max_length(self):
        return np.linalg.norm(self._get_bounding_box())

    line_with_factor = Float(0.004)

    def _get_line_width(self):
        return self._get_max_length() * self.line_with_factor

    def plot_mlab(self, mlab, nodes=True, lines=True, L_selection=[]):
        r'''Visualize the crease pattern in a supplied mlab instance.
        '''
        x, y, z = self.x.T
        if len(self.F) > 0:
            cp_pipe = mlab.triangular_mesh(x, y, z, self.F,
                                           line_width=3,
                                           color=(0.6, 0.6, 0.6))
            if lines is True:
                L = self.L
                if len(L_selection):
                    L = L[L_selection]
                cp_pipe.mlab_source.dataset.lines = L
                tube = mlab.pipeline.tube(cp_pipe,
                                          tube_radius=self._get_line_width())
                mlab.pipeline.surface(tube, color=(1.0, 1.0, 1.0))

        else:
            cp_pipe = mlab.points3d(x, y, z, scale_factor=0.2)
            cp_pipe.mlab_source.dataset.lines = self.L
        return cp_pipe.mlab_source

    viz3d_classes = dict(cp=CreasePatternViz3D,
                         node_numbers=CreasePatternNodeNumbersViz3D,
                         displ=CreasePatternDisplViz3D)

    def get_cnstr_pos(self, iteration_step):
        r'''
         Get the coordinates of the constraints.

        @todo this should be moved to GuDofConstraints
        '''
        print('get position')
        u_t = self.fold_steps[iteration_step]
        pts_p, faces_p = self.cnstr[0].get_cnstr_view(u_t, 1.0)
        pts_l = None
        con_l = None
        return (pts_l, con_l, pts_p, faces_p)

    # =========================================================================
    # Garbage
    # =========================================================================
    def get_line_position(self, i):
        r'''
        This method prints the procentual position of a linepoint element on
        his line over all timesteps.

        i [int]: This value represents the index of a linepoint element,
                 which should be reviewed.
        '''

        if(len(self.line_pts) == 0):
            print(' NO LINE POINTS')
            return

        for p in range(len(self.fold_steps)):
            cl = self.crease_lines[self.line_pts[i][1]]
            p1 = self.fold_steps[p][cl[0]]
            p2 = self.fold_steps[p][cl[1]]
            p0 = self.fold_steps[p][self.line_pts[i][0]]

            try:
                rx = (p0[0] - p1[0]) / (p2[0] - p1[0])
            except:
                rx = 0
            try:
                ry = (p0[1] - p1[1]) / (p2[1] - p1[1])
            except:
                ry = 0
            try:
                rz = (p0[2] - p1[2]) / (p2[2] - p1[2])
            except:
                rz = 0

            if(rx != 0):
                r = rx
            elif (ry != 0):
                r = ry
            else:
                r = rz
            print('Step ', p, ': r = ', r)

    def create_rcp_tex(self, name='rcp_output.tex', x=15., y=15.):
        r'''
        This methode returns a *.tex file with the top view of the
        creasepattern and the nodeindex of every node. This file
        can be implemented into a latex documentation, using package
        pst-all.
        '''
        n = self.X
        c = self.L
        x_l = np.max(n[:, 0])
        y_l = np.max(n[:, 1])
        x_size = x / x_l
        y_size = x / y_l
        if(x_size < y_size):
            size = x_size
        else:
            size = y_size
        f = open(name, 'w')
        f.write('\\psset{xunit=%.3fcm,yunit=%.3fcm}\n' % (size, size))
        f.write(' \\begin{pspicture}(0,%.3f)\n' % (y_l))
        for i in range(len(n)):
            if(n[i][2] == 0):
                f.write('  \\cnodeput(%.3f,%.3f){%s}{\\footnotesize%s}\n' %
                        (n[i][0], n[i][1], i, i))
        for i in range(len(c)):
            if(n[c[i][0]][2] == 0 and n[c[i][1]][2] == 0):
                f.write('  \\ncline{%s}{%s}\n' % (c[i][0], c[i][1]))
        f.write(' \\end{pspicture}' + '\n')
        f.close()

    def create_3D_tex(self, name='standart3Doutput.tex', x=5, y=5,
                      alpha=140, beta=30):
        r'''
        This method returns a .tex file with a 3D view of the
        creasepattern and the nodeindex of every node, as a sketch. This file
        can be implemented into a latex documentation, using package
        pst-3dplot.
        '''
        n = self.X
        c = self.L
        f = open(name, 'w')
        # f.write('\\configure[pdfgraphic][width=%.3f,height=%.3f]\n' %(x, y))
        # f.write('\\begin{pdfdisplay}\n')
        f.write('\\psset{xunit=%.3fcm,yunit=%.3fcm,Alpha=%.3f,Beta=%.3f}\n' %
                (x, y, alpha, beta))
        f.write(' \\begin{pspicture}(0,0)\n')
        f.write(' \\pstThreeDCoor\n')
        for i in range(len(n)):
            f.write('  \\pstThreeDNode(%.3f,%.3f,%.3f){%s}\n' %
                    (n[i][0], n[i][1], n[i][2], i))
        for i in range(len(c)):
            if(n[c[i][0]][2] == 0 and n[c[i][1]][2] == 0):
                f.write(' \\psset{dotstyle=*,linecolor=gray}\n')
            else:
                f.write(' \\psset{linecolor=black}\n')
            f.write('  \\pstThreeDLine(%.3f,%.3f,%.3f)(%.3f,%.3f,%.3f)\n' %
                    (n[c[i][0]][0], n[c[i][0]][1], n[c[i][0]][2],
                     n[c[i][1]][0], n[c[i][1]][1], n[c[i][1]][2]))
        f.write(' \\psset{dotstyle=*,linecolor=gray}\n')
        for i in range(len(n)):
            f.write('  \\pstThreeDDot(%.3f,%.3f,%.3f)\n' %
                    (n[i][0], n[i][1], n[i][2]))
        f.write(' \\psset{linecolor=black}\n')
        for i in range(len(n)):
            f.write('  \\pstThreeDPut(%.3f,%.3f,%.3f){%s}\n' %
                    (n[i][0], n[i][1], n[i][2], i))
        f.write(' \\end{pspicture}' + '\n')
#        f.write(' \\end{pdfdisplay}' + '\n')
        f.close()
