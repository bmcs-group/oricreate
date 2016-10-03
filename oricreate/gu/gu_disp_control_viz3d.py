'''
Created on Dec 3, 2015

@author: rch
'''

import copy
from traits.api import \
    Property, Float, Color, \
    cached_property

import numpy as np
from oricreate.viz3d import Viz3D


class GuDofConstraintsViz3D(Viz3D):
    '''Visualize the crease Pattern
    '''

    scale_factor = Float(1.0, auto_set=False, enter_set=True)

    cnstr = Property(depends_on='vis3d')

    @cached_property
    def _get_cnstr(self):
        '''
        This property prepares the constraints for the visualization

        It extracts the information from the current  crease - pattern
        and divides it for easier calculation of the symbol - positions

        The constrains are divided in three constrain - types:
        - fixed constrains (constrain is full fixed in his direction)
        - connected constrains (constrains are in depending on each other
                           , e.g. constant or linear movement behavior)
        - load constrains (constrains, which activates the numerical calculation)

        Different list for visualization:
        fixed constrains:
        - cn_f : index array of nodes of fixed constrains
        - cd_f : direction (axes) of the fixed constrain [x, y, z]

        connected constrains:
        - cn_c: index array of nodes of fixed constrains
        - cc_c: connection of nodes as index arrays, e.g. [[0, 1],
                                                      [2, 4]]
            each index represents a node
        - cd_c: direction (axes) of the connected constrains [x, y, z]

        load constrains:
        - cn_l : index array of nodes of load constrains
        - cd_l : direction (axes) of the load constrain [x, y, z]
        '''
        # get constrain information of the crease pattern

        lhs_list = []
        rhs_list = []
        for dof_constraint in self.vis3d.dof_constraints:
            lhs_entry, rhs_entry = dof_constraint
            lhs_list.append(lhs_entry)
            rhs_list.append(rhs_entry)

        # get load constrains

        load_nodes = np.array([], dtype=int)
        load_dir = np.array([])
        count = 0

        lhs_copy = copy.deepcopy(lhs_list)
        while(count < len(rhs_list)):
            # all cell's in rhs which are different 0 represents a load and
            # gives the direction
            # the constrain on the same indexposition in lhs is the load
            # constrain
            if (rhs_list[count] != None):
                node = lhs_list[count][0][0]
                dir_vec = np.array([0, 0, 0])
                dir_vec[lhs_list[count][0][1]] = 1

                if(rhs_list[count](1.0) < 0):
                    dir_vec *= -1

                load_nodes = np.append(load_nodes, node)
                load_dir = np.append(load_dir, [dir_vec])
                # remove the constrain from lhs-list
                lhs_copy.remove(lhs_list[count])

            count += 1

        lhs = copy.deepcopy(lhs_copy)
        load_nodes = load_nodes.reshape(len(load_nodes), 1)
        load_dir = load_dir.reshape(len(load_nodes), 3)

        # divide all other constrains to fixed or connected constrains

        cnstr_fixed = lhs
        cnstr_connect = []

        count = 0
        while(count < len(cnstr_fixed)):
            # put all connected constrains out of cnstr_fixed into
            # cnstr_connect
            if len(cnstr_fixed[count]) > 1:
                cnstr_connect.append(cnstr_fixed.pop(count))
                continue

            count += 1

        # create Cnstr Arrays

        fixed_nodes = np.array([], dtype=int)
        fixed_dir = np.array([])

        # build cn_f and cd_fopacity_min = Int(0)

        for i in cnstr_fixed:
            fixed_nodes = np.append([fixed_nodes], [i[0][0]])
            dir_vec = np.array([0, 0, 0])
            dir_vec[i[0][1]] = 1
            fixed_dir = np.append([fixed_dir], [dir_vec])

        fixed_nodes = fixed_nodes.reshape(len(cnstr_fixed), 1)
        fixed_dir = fixed_dir.reshape(len(cnstr_fixed), 3)

        # get connections on real node indexes

        c_nodes = np.array([], dtype=int)
        c_dir = np.array([])
        con = np.array([], dtype=int)
        count = 0
        for i in cnstr_connect:
            c_nodes = np.append(c_nodes, i[0][0])
            c_nodes = np.append(c_nodes, i[1][0])
            vct1 = np.zeros((3,))
            vct2 = np.zeros((3,))
            vct1[i[0][1]] = 1
            vct2[i[1][1]] = 1
            c_dir = np.append(c_dir, vct1)
            c_dir = np.append(c_dir, vct2)
            c = np.array([count, count + 1])
            con = np.append(con, c)
            count += 2

        c_dir = c_dir.reshape((-1, 3))
        con = con.reshape((-1, 2))

        return (fixed_nodes, fixed_dir, c_nodes, con, c_dir,
                load_nodes, load_dir)

    min_max = Property
    '''Rectangular bounding box. 
    '''

    def _get_min_max(self):
        vis3d = self.vis3d
        return np.min(vis3d.x, axis=0), np.max(vis3d.x, axis=0)

    facet_color = Color((0.4, 0.4, 0.7))
    facet_color = Color((0.0 / 255.0, 84.0 / 255.0, 159.0 / 255.0))

    def plot(self):

        m = self.ftv.mlab
        cp = self.vis3d.formed_object
        x_t = cp.x

        cn_f, cd_f, cn_c, cc_c, cd_c, cn_l, cd_l = self.cnstr

        # spacefactor is giving space between constrains an real node
        # position

        spacefactor = 0.02 * self.scale_factor
        scale = self.scale_factor * 1.0
        line_width = self.scale_factor * 2
        # fixed cnstr
        cp_f = x_t[cn_f]
        x, y, z = cp_f.T
        x, y, z = x[0], y[0], z[0]
        U, V, W = cd_f.T * scale
        sU, sV, sW = cd_f.T * spacefactor

        x = x - U - sU
        y = y - V - sV
        z = z - W - sW

        cf_arrow = m.quiver3d(x, y, z, U, V, W,
                              mode='2darrow', color=(0.0, 0.0, 1.0),
                              scale_mode='vector', scale_factor=self.scale_factor,
                              line_width=line_width)
        cf_cross = m.quiver3d(x, y, z, U, V, W, mode='2dcross',
                              color=(0.0, 0.0, 1.0),
                              scale_mode='vector',
                              scale_factor=self.scale_factor,
                              line_width=line_width)

        m.pipeline.surface(cf_cross)
        m.pipeline.surface(cf_arrow)

        self.pipes['cf_cross'] = cf_cross
        self.pipes['cf_arrow'] = cf_arrow
        # load constraint

        cp_l = x_t[cn_l]

        x, y, z = cp_l.T
        x, y, z = x[0], y[0], z[0]
        U, V, W = cd_l.T * scale
        sU, sV, sW = cd_l.T * spacefactor

        x = x - U - sU
        y = y - V - sV
        z = z - W - sW

        cl_arrow = m.quiver3d(x, y, z, U, V, W, mode='arrow',
                              color=(1.0, 0.0, 0.0),
                              scale_mode='vector',
                              line_width=line_width,
                              scale_factor=self.scale_factor)
        m.pipeline.surface(cl_arrow)
        self.pipes['cl_arrow'] = cl_arrow

        # connected constraints

        cp_c = x_t[cn_c]

        x, y, z = cp_c.T

        U, V, W = cd_c.T * scale
        sU, sV, sW = cd_c.T * spacefactor

        x = x - U - sU
        y = y - V - sV
        z = z - W - sW

        cc_arrow = m.quiver3d(x, y, z, U, V, W,
                              mode='2darrow',
                              line_width=line_width,
                              color=(0.0, 1.0, 0.0),
                              scale_mode='vector',
                              scale_factor=self.scale_factor)

        cc_arrow.mlab_source.dataset.lines = cc_c

        m.pipeline.surface(cc_arrow, color=(0.0, 0.7, 0.0),
                           line_width=line_width)
        self.pipes['cc_arrow'] = cc_arrow

    def update(self):

        cp = self.vis3d.formed_object
        x_t = cp.x

        # update constrain symbols

        cn_f, cd_f, cn_c, cc_c, cd_c, cn_l, cd_l = self.cnstr

        spacefactor = 0.02 * self.scale_factor
        scale = self.scale_factor * 0.5

        # fixed cnstr
        cp_f = x_t[cn_f]
        x, y, z = cp_f.T
        x, y, z = x[0], y[0], z[0]
        U, V, W = cd_f.T * scale
        sU, sV, sW = cd_f.T * spacefactor

        x = x - U - sU
        y = y - V - sV
        z = z - W - sW

        cf_arrow = self.pipes['cf_arrow']
        cf_arrow.mlab_source.reset(x=x, y=y, z=z)
        cf_cross = self.pipes['cf_cross']
        cf_cross.mlab_source.reset(x=x, y=y, z=z)

        # load constrains
        cp_l = x_t[cn_l]

        x, y, z = cp_l.T
        x, y, z = x[0], y[0], z[0]
        U, V, W = cd_l.T * scale
        sU, sV, sW = cd_l.T * spacefactor

        x = x - U - sU
        y = y - V - sV
        z = z - W - sW

        cl_arrow = self.pipes['cl_arrow']
        cl_arrow.mlab_source.reset(x=x, y=y, z=z)

        # connected constrains
        cp_c = x_t[cn_c]

        x, y, z = cp_c.T

        U, V, W = cd_c.T * scale
        sU, sV, sW = cd_c.T * spacefactor

        x = x - U - sU
        y = y - V - sV
        z = z - W - sW

        cc_arrow = self.pipes['cc_arrow']
        cc_arrow.mlab_source.reset(x=x, y=y, z=z)
        cc_arrow.mlab_source.dataset.lines = cc_c

    def _get_bounding_box(self):
        cp = self.vis3d.formed_object
        x_t = cp.x
        return np.min(x_t, axis=0), np.max(x_t, axis=0)

    def _get_max_length(self):
        return np.linalg.norm(self._get_bounding_box())
