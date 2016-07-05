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


from traits.api import \
    implements,  List, Tuple, Float, \
    cached_property, Property, Array, Int

from fu import \
    Fu
from fu_poteng_bending_viz3d import \
    FuPotEngBendingViz3D
from fu_poteng_node_load_viz3d import \
    FuPotEngNodeLoadViz3D
import numpy as np
from oricreate.opt import \
    IFu
from oricreate.util.einsum_utils import \
    EPS
from oricreate.viz3d import \
    Visual3D


class FuPotEngTotal(Fu, Visual3D):

    '''Optimization criteria based on minimum Bending energy of gravity.

    This plug-in class lets the crease pattern operators evaluate the
    integral over the spatial domain in an instantaneous configuration
    '''

    implements(IFu)

    F_ext_list = List(Tuple, [])

    _kappa_arr = Array(float, value=[])
    _kappa = Float(1.0)

    kappa = Property()

    debug_level = Int(0, auto_set=False, enter_set=True)

    def _set_kappa(self, value):
        if isinstance(value, np.ndarray):
            self._kappa_arr = value
        elif isinstance(value, float) or isinstance(value, int):
            self._kappa = value
        else:
            raise ValueError, 'wrong type of kappa value - must be float or array'

    def _get_kappa(self):
        if len(self._kappa_arr):
            return self._kappa_arr
        else:
            cp = self.forming_task.formed_object
            return np.ones((cp.n_iL,), dtype='float_') * self._kappa

    fu_factor = Float(1.0)

    def get_f(self, t=0):
        '''Get the total potential energy.
        '''
        cp = self.forming_task.formed_object
        iL_phi = cp.iL_psi2 - cp.iL_psi_0
        iL_length = np.linalg.norm(cp.iL_vectors, axis=1)

        stored_energy = np.einsum(
            '...i,...i,...i->...', self.kappa, iL_phi**2, iL_length) / 2.0

        F_ext = np.zeros_like(cp.u, dtype='float_')
        for node, dim, value in self.F_ext_list:
            F_ext[node, dim] = value

        ext_energy = np.einsum(
            '...i,...i->...', F_ext.flatten(), cp.u.flatten())
        tot_energy = self.fu_factor * (stored_energy - ext_energy)
        print 'tot_energy', tot_energy
        return tot_energy

    def get_f_du(self, t=0):
        '''Get the derivatives with respect to individual displacements.
        '''
        cp = self.forming_task.formed_object

        vl = cp.iL_vectors
        nl0, nl1 = np.einsum('fi...->if...', cp.iL_F_normals)
        if self.debug_level > 0:
            print 'vl', vl.shape
            print vl
            print 'nl0', nl0.shape
            print nl0
            print 'nl1', nl1.shape
            print nl1

        # construct the unit orthonormal basis
        norm_vl = np.sqrt(np.einsum('...i,...i->...', vl, vl))
        norm_nl0 = np.sqrt(np.einsum('...i,...i->...', nl0, nl0))
        norm_nl1 = np.sqrt(np.einsum('...i,...i->...', nl1, nl1))
        unit_vl = vl / norm_vl[:, np.newaxis]
        unit_nl0 = nl0 / norm_nl0[:, np.newaxis]
        unit_nl1 = nl1 / norm_nl1[:, np.newaxis]
        if self.debug_level > 0:
            print 'unit_vl', unit_vl.shape
            print unit_vl
            print 'unit_nl0', unit_nl0.shape
            print unit_nl0
            print 'unit_nl1', unit_nl1.shape
            print unit_nl1

        # construct transformation matrix
        Tl0 = np.einsum('ij...->ji...',
                        np.array(
                            [unit_vl,
                             unit_nl0,
                             np.einsum('...j,...k,...ijk->...i',
                                       unit_vl, unit_nl0, EPS)]
                        ))
        if self.debug_level > 0:
            print 'Tl0', Tl0.shape
            print Tl0

        unit_nl01 = np.einsum('...ij,...j->...i', Tl0, unit_nl1)
        if self.debug_level > 0:
            print 'unit_nl01[:,2]', unit_nl01[:, 2]
            print unit_nl01[:, 2]

        psi = np.arcsin(unit_nl01[:, 2])
        if self.debug_level > 0:
            print 'psi', psi

        vl_dul = cp.iL_vectors_dul
        nl0_dul0, nl1_dul1 = np.einsum('fi...->if...', cp.iL_F_normals_du)
        if self.debug_level > 0:
            print cp.iL_N.shape
            print 'vl_dul', vl_dul.shape
            print vl_dul
            print 'nl0_dul0', nl0_dul0.shape
            print nl0_dul0
            print 'nl1_dul1', nl1_dul1.shape
            print nl1_dul1

        unit_nl0_dul0 = 1 / norm_nl0[:, np.newaxis, np.newaxis, np.newaxis] * (
            nl0_dul0 -
            np.einsum('...j,...i,...iNd->...jNd', unit_nl0, unit_nl0, nl0_dul0)
        )
        unit_nl1_dul1 = 1 / norm_nl1[:, np.newaxis, np.newaxis, np.newaxis] * (
            nl1_dul1 -
            np.einsum('...j,...i,...iNd->...jNd', unit_nl1, unit_nl1, nl1_dul1)
        )
        unit_vl_dul = 1 / norm_vl[:, np.newaxis, np.newaxis, np.newaxis] * (
            vl_dul -
            np.einsum('...j,...i,...iNd->...jNd', unit_vl, unit_vl, vl_dul)
        )
        if self.debug_level > 0:
            print 'unit_nl0_dul0', unit_nl0_dul0.shape
            print unit_nl0_dul0
            print 'unit_nl1_dul1', unit_nl1_dul1.shape
            print unit_nl1_dul1
            print 'unit_vl_dul', unit_vl_dul.shape
            print unit_vl_dul

        Tl0_dul0 = np.einsum('ij...->ji...',
                             np.array([np.zeros_like(unit_nl0_dul0),
                                       unit_nl0_dul0,
                                       np.einsum(
                                      '...j,...kNd,...ijk->...iNd',
                                      unit_vl, unit_nl0_dul0, EPS)
                             ]
                             ))

        if self.debug_level > 0:
            print 'Tl0_dul0', Tl0_dul0.shape
            print Tl0_dul0

        Tl0_dul = np.einsum('ij...->ji...',
                            np.array([unit_vl_dul,
                                      np.zeros_like(unit_vl_dul),
                                      np.einsum(
                                          '...jNd,...k,...ijk->...iNd',
                                          unit_vl_dul, unit_nl0, EPS)
                                      ]
                                     )
                            )
        if self.debug_level > 0:
            print 'Tl0_dul0', Tl0_dul.shape
            print Tl0_dul

        rho = 1 / np.sqrt((1 - unit_nl01[:, 2]**2))
        if self.debug_level > 0:
            print 'rho', unit_nl01[:, 2]

        unit_nl01_dul = np.einsum(
            '...,...j,...ijNd->...iNd', rho, unit_nl1, Tl0_dul)[:, 2, ...]
        unit_nl01_dul0 = np.einsum(
            '...,...j,...ijNd->...iNd', rho, unit_nl1, Tl0_dul0)[:, 2, ...]
        unit_nl01_dul1 = np.einsum(
            '...,...jNd,...ij->...iNd', rho, unit_nl1_dul1, Tl0)[:, 2, ...]
        if self.debug_level > 0:
            print 'unit_nl01_dul', unit_nl01_dul.shape
            print unit_nl01_dul
            print 'unit_nl01_dul0', unit_nl01_dul0.shape
            print unit_nl01_dul0
            print 'unit_nl01_dul1', unit_nl01_dul1.shape
            print unit_nl01_dul1

        # get the map of facet nodes attached to interior lines
        iL0_N_map = cp.F_N[cp.iL_F[:, 0]].reshape(cp.n_iL, -1)
        iL1_N_map = cp.F_N[cp.iL_F[:, 1]].reshape(cp.n_iL, -1)
        iL_N_map = cp.F_L_N[cp.iL_within_F0]

        # enumerate the interior lines and broadcast it N and D into dimensions
        iL_map = np.arange(cp.n_iL)[:, np.newaxis, np.newaxis]
        # broadcast the facet node map into D dimension
        l0_map = iL0_N_map[:, :, np.newaxis]
        l1_map = iL1_N_map[:, :, np.newaxis]
        l_map = iL_N_map[:, :, np.newaxis]
        # broadcast the spatial dimension map into iL and N dimensions
        D_map = np.arange(3)[np.newaxis, np.newaxis, :]
        # allocate the gamma derivatives of iL with respect to N and D
        # dimensions
        psi_du = np.zeros((cp.n_iL, cp.n_N, cp.n_D), dtype='float_')
        # add the contributions gamma_du from the left and right facet
        # Note: this cannot be done in a single step since the incremental
        # assembly is not possible within a single index expression.
        psi_du[iL_map, l_map, D_map] += unit_nl01_dul
        if self.debug_level > 0:
            print 'l_map', l_map.shape
            print l_map
            print 'psi_du', psi_du.shape
            print psi_du
        psi_du[iL_map, l0_map, D_map] += unit_nl01_dul0
        if self.debug_level > 0:
            print 'l0_map', l0_map.shape
            print l0_map
            print 'psi_du', psi_du.shape
            print psi_du
        psi_du[iL_map, l1_map, D_map] += unit_nl01_dul1
        if self.debug_level > 0:
            print 'l1_map', l1_map.shape
            print l1_map
            print 'psi_du', psi_du.shape
            print psi_du

        F_ext = np.zeros_like(cp.u, dtype='float_')
        for node, dim, value in self.F_ext_list:
            F_ext[node, dim] = value

        iL_phi = cp.iL_psi2 - cp.iL_psi_0
        iL_length = np.linalg.norm(cp.iL_vectors, axis=1)

        Pi_int_du = np.einsum('...l,...l,...l,...lId->...Id',
                              iL_length, self.kappa, iL_phi, psi_du)

        Pi_ext_du = F_ext

        Pi_du = Pi_int_du - Pi_ext_du

        return Pi_du.flatten()

    viz3d_dict = Property

    @cached_property
    def _get_viz3d_dict(self):
        return dict(default=FuPotEngBendingViz3D(vis3d=self),
                    node_load=FuPotEngNodeLoadViz3D(vis3d=self))
