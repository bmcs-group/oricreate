# -------------------------------------------------------------------------------
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
# Created on Jan 29, 2015 by: rch

from traits.api import \
    HasStrictTraits, Float, \
    Property, cached_property, \
    Array

import numpy as np
from oricreate.util import \
    get_theta, get_theta_du
from oricreate.util.einsum_utils import \
    DELTA, EPS


INPUT = '+cp_input'


class CreaseNodeOperators(HasStrictTraits):

    r'''Operators delivering the instantaneous values of parameters
    related to nodes. All attributes return arrays with first dimension
    equal to ``n_N``.
    '''
    iN_theta = Property
    r'''List of arrays with sector angles around each interior node
    in the format
    ``[ np.array([neighbor_node1, neighbor_node2, ... neighbor_node1), ... ]``
    '''

    def _get_iN_theta(self):
        NN_theta = self.NN_theta
        return [NN_theta[n, neighbors[:-1]]
                for n, neighbors in zip(self.iN, self.iN_nbr)]

    iN_theta_du = Property
    r'''Assemble the derivatives of sector angles around a node :math:`i`
    '''

    def _get_iN_theta_du(self):
        NN_theta_du = self.NN_theta_du
        return [NN_theta_du[n, neighbors[:-1], :, :]
                for n, neighbors in zip(self.iN, self.iN_nbr)]

    NN_theta = Property
    r'''Matrix of angles ``[n_L,n_L]`` containing the values of sector angle
    between two consecutive lines. Angles are provided for each
    pair of lines that fulfill the following condition:
    If the first line connects the nodes :math:`ij` the second can
    be obtained by rotating the first line by the sector angle
    :math:`\\theta_{ij}` around the node :math:`i` in a counter-
    clockwise manner. Remaining position in the matrix are set to zero.

    This matrix serves as an interim result for a simple assembly
    of the ``iN_theta`` array.

    .. image:: figs/crease_node_operators_NN_theta.png
    '''

    def _get_NN_theta(self):
        NN_theta = np.zeros_like(self.NN_L, dtype='float_')
        F_theta = self.F_theta
        NN_theta[self.F_N[:, (0, 1, 2)], self.F_N[:, (1, 2, 0)]] = \
            F_theta[:, (0, 1, 2)]
        return NN_theta

    NN_theta_du = Property
    r'''Array of derivatives ``[n_L,n_L,n_dofs]`` of an angle between
    two lines of a facet.
    '''

    def _get_NN_theta_du(self):
        NN_theta_du = np.zeros((self.n_N, self.n_N, self.n_N, self.n_D),
                               dtype='float_')
        F_theta_du = self.F_theta_du
        NN_theta_du[self.F_N[:, (0, 1, 2)], self.F_N[:, (1, 2, 0)], ...] = \
            F_theta_du[:, (0, 1, 2), ...]
        return NN_theta_du


class CreaseLineOperators(HasStrictTraits):

    r'''Operators delivering the instantaneous states of crease lines.
    '''

    # =========================================================================
    # Property operators for initial configuration
    # =========================================================================
    L_vectors_0 = Property(Array, depends_on=INPUT)
    r'''Vectors of the crease lines.

    ... math::
        \bm{v} = \bm{x}_2 - \bm{x}_1

    '''
    @cached_property
    def _get_L_vectors_0(self):
        return self.x_0[self.L[:, 1], :] - self.x_0[self.L[:, 0], :]

    L_vectors = Property(Array, depends_on=INPUT)
    r'''Vectors of the crease lines.

    ... math::
        \bm{v} = \bm{x}_2 - \bm{x}_1

    '''
    @cached_property
    def _get_L_vectors(self):
        return self.x[self.L[:, 1]] - self.x[self.L[:, 0]]

    L_lengths = Property(Array, depends_on=INPUT)
    r'''Lengths of the crease lines.

    ... math::
        \ell = \left\| \bm{v} \right\|

    '''
    @cached_property
    def _get_L_lengths(self):
        v = self.L_vectors
        return np.sqrt(np.sum(v ** 2, axis=1))

    L_vectors_du = Property(Array, depends_on=INPUT)
    r'''Get the derivatives of the line vectors

    .. math::
       v_{l,d,\mathrm{lnode}(l,0),e} = -\delta_{de}

       v_{l,d,\mathrm{lnode}(l,1),e} = +\delta_{de}

       v_{l,d,I,e} = 0 \; \mathrm{where} \; I \neq \mathrm{lnode}(l,[0,1])

    with the indices :math:`l,d,I,e` representing the line, component,
    node and coordinate, respectively.
    '''
    @cached_property
    def _get_L_vectors_du(self):
        L_vectors_du = np.zeros((self.n_L, self.n_D, self.n_N, self.n_D),
                                dtype='float_')
        L_idx = np.arange(self.n_L)
        L_N0_idx = self.L[L_idx, 0]
        L_N1_idx = self.L[L_idx, 1]
        L_vectors_du[L_idx, :, L_N0_idx, :] = -DELTA
        L_vectors_du[L_idx, :, L_N1_idx, :] = DELTA
        return L_vectors_du

    L_vectors_dul = Property(Array, depends_on=INPUT)

    @cached_property
    def _get_L_vectors_dul(self):
        L_vectors_du = np.zeros((self.n_L, self.n_D, 2, self.n_D),
                                dtype='float_')
        L_idx = np.arange(self.n_L)
        L_vectors_du[L_idx, :, 0, :] = -DELTA
        L_vectors_du[L_idx, :, 1, :] = DELTA
        return L_vectors_du

    iL_within_F0 = Property(Array, depends_on=INPUT)
    r'''Index of a crease line within the first adjacent facet
    '''
    @cached_property
    def _get_iL_within_F0(self):
        iL_F0 = self.iL_F[:, 0]
        L_of_F0_of_iL = self.F_L[iL_F0, :]
        iL = self.iL
        #iL_within_F0 = np.where(iL[:, np.newaxis] == L_of_F0_of_iL)
        L_within_F0, ell_within_F0 = np.where(
            iL[:, np.newaxis] == L_of_F0_of_iL)
        F0_iL = (iL_F0[L_within_F0], ell_within_F0)
        return F0_iL

    iL_vectors = Property(Array, depends_on=INPUT)
    r'''Get the line vector of an interior line oriented in the
    sense of counter-clockwise direction of its first adjacent facet.
    '''
    @cached_property
    def _get_iL_vectors(self):
        F_L_vectors = self.F_L_vectors
        return F_L_vectors[self.iL_within_F0]

    iL_vectors_du = Property(Array, depends_on=INPUT)
    r'''Get the line vector of an interior line oriented in the
    sense of counter-clockwise direction of its first adjacent facet.
    '''
    @cached_property
    def _get_iL_vectors_du(self):
        F_L_vectors_du = self.F_L_vectors_du
        return F_L_vectors_du[self.iL_within_F0]

    iL_vectors_dul = Property(Array, depends_on=INPUT)
    r'''Get the derivatives with respect to the node coordinates 
    of a  line vector of an interior line oriented in the
    sense of counter-clockwise direction of its first adjacent facet.
    '''
    @cached_property
    def _get_iL_vectors_dul(self):
        F_L_vectors_dul = self.F_L_vectors_dul
        return F_L_vectors_dul[self.iL_within_F0]

    iL_vectors_0 = Property(Array, depends_on=INPUT)
    r'''Get the line vector of an interior line oriented in the
    sense of counter-clockwise direction of its first adjacent facet.
    '''
    @cached_property
    def _get_iL_vectors_0(self):
        F_L_vectors = self.F_L_vectors_0
        return F_L_vectors[self.iL_within_F0]

    norm_iL_vectors = Property(Array, depends_on=INPUT)
    r'''Get the normed line vector of an interior line oriented in the
    sense of counter-clockwise direction of its first adjacent facet.
    '''

    def _get_norm_iL_vectors(self):
        iL_vectors = self.iL_vectors
        mag_iL_vectors = np.sqrt(np.einsum('...i,...i->...',
                                           iL_vectors, iL_vectors))
        return iL_vectors / mag_iL_vectors[:, np.newaxis]

    norm_iL_vectors_0 = Property(Array, depends_on=INPUT)
    r'''Get the normed line vector of an interior line oriented in the
    sense of counter-clockwise direction of its first adjacent facet.
    '''

    def _get_norm_iL_vectors_0(self):
        iL_vectors = self.iL_vectors_0
        mag_iL_vectors = np.sqrt(np.einsum('...i,...i->...',
                                           iL_vectors, iL_vectors))
        return iL_vectors / mag_iL_vectors[:, np.newaxis]

    iL_F_normals = Property(Array, depends_on=INPUT)
    r'''Get normals of facets adjacent to an interior line.
    '''
    @cached_property
    def _get_iL_F_normals(self):
        F_normals = self.F_normals
        return F_normals[self.iL_F]

    iL_F_normals_du = Property(Array, depends_on=INPUT)
    r'''Get normals of facets adjacent to an interior line.
    '''
    @cached_property
    def _get_iL_F_normals_du(self):
        F_normals_du = self.F_normals_du
        return F_normals_du[self.iL_F]

    iL_F_normals_0 = Property(Array, depends_on=INPUT)
    r'''Get normals of facets adjacent to an interior line.
    '''
    @cached_property
    def _get_iL_F_normals_0(self):
        F_normals = self.F_normals_0
        return F_normals[self.iL_F]

    norm_iL_F_normals = Property(Array, depends_on=INPUT)
    r'''Get normed normals of facets adjacent to an interior line.
    '''
    @cached_property
    def _get_norm_iL_F_normals(self):
        norm_F_normals = self.norm_F_normals
        return norm_F_normals[self.iL_F]

    norm_iL_F_normals_0 = Property(Array, depends_on=INPUT)
    r'''Get normed normals of facets adjacent to an interior line.
    '''
    @cached_property
    def _get_norm_iL_F_normals_0(self):
        norm_F_normals = self.norm_F_normals_0
        return norm_F_normals[self.iL_F]

    iL_psi = Property(Array, depends_on=INPUT)
    r'''Calculate the dihedral angle for the intermediate configuration.
    
    psi + 2*(pi/2 - psi) 
    '''
    @cached_property
    def _get_iL_psi(self):
        l = self.norm_iL_vectors
        n = self.norm_iL_F_normals
        n0, n1 = np.einsum('ijk->jik', n)
        p0 = np.einsum('...i,...j,...kij->...k', l, n0, EPS)
        T = np.concatenate([l[:, np.newaxis, :],
                            n0[:, np.newaxis, :],
                            p0[:, np.newaxis, :]], axis=1)
        n1_ = np.einsum('...ij,...j->...i', T, n1)
        n1_1, n1_2 = n1_[:, 1:].T
        psi = np.arcsin(n1_2)
        oa_idx = np.where(n1_1 < 0.0)[0]
        shifted_psi = np.sign(n1_2[oa_idx]) * np.pi - 2.0 * psi[oa_idx]
        psi[oa_idx] += shifted_psi
        return psi

    iL_psi_0 = Property(Array, depends_on=INPUT)
    r'''Calculate the dihedral angle for the intermediate configuration.
    '''
    @cached_property
    def _get_iL_psi_0(self):
        l = self.norm_iL_vectors_0
        n = self.norm_iL_F_normals_0
        n0, n1 = np.einsum('ijk->jik', n)
        p0 = np.einsum('...i,...j,...kij->...k', l, n0, EPS)
        T = np.concatenate([l[:, np.newaxis, :],
                            n0[:, np.newaxis, :],
                            p0[:, np.newaxis, :]], axis=1)
        n1_ = np.einsum('...ij,...j->...i', T, n1)
        n1_cos, n1_sin = n1_[:, 1:].T
        oa_idx = np.where(n1_cos < 0.0)[0]
        psi = np.arcsin(n1_sin)
        shifted_psi = np.sign(n1_sin[oa_idx]) * np.pi - 2.0 * psi[oa_idx]
        psi[oa_idx] += shifted_psi
        return psi

    iL_psi_du = Property(Array, depends_on=INPUT)
    r'''Calculate the derivatives of the dihedral angle 
    for the intermediate configuration.
    '''
    @cached_property
    def _get_iL_psi_du(self, t=0):
        '''Get the derivatives with respect to individual displacements.
        '''
        vl = self.iL_vectors
        nl0, nl1 = np.einsum('fi...->if...', self.iL_F_normals)
        if self.debug_level > 0:
            print('vl', vl.shape)
            print(vl)
            print('nl0', nl0.shape)
            print(nl0)
            print('nl1', nl1.shape)
            print(nl1)

        # construct the unit orthonormal basis
        norm_vl = np.sqrt(np.einsum('...i,...i->...', vl, vl))
        norm_nl0 = np.sqrt(np.einsum('...i,...i->...', nl0, nl0))
        norm_nl1 = np.sqrt(np.einsum('...i,...i->...', nl1, nl1))
        unit_vl = vl / norm_vl[:, np.newaxis]
        unit_nl0 = nl0 / norm_nl0[:, np.newaxis]
        unit_nl1 = nl1 / norm_nl1[:, np.newaxis]
        if self.debug_level > 0:
            print('unit_vl', unit_vl.shape)
            print(unit_vl)
            print('unit_nl0', unit_nl0.shape)
            print(unit_nl0)
            print('unit_nl1', unit_nl1.shape)
            print(unit_nl1)

        # construct transformation matrix
        Tl0 = np.einsum('ij...->ji...',
                        np.array(
                            [unit_vl,
                             unit_nl0,
                             np.einsum('...j,...k,...ijk->...i',
                                       unit_vl, unit_nl0, EPS)]
                        ))
        if self.debug_level > 0:
            print('Tl0', Tl0.shape)
            print(Tl0)

        unit_nl01 = np.einsum('...ij,...j->...i', Tl0, unit_nl1)
        if self.debug_level > 0:
            print('unit_nl01[:,2]', unit_nl01[:, 2])
            print(unit_nl01[:, 2])

        n1_cos, n1_sin = unit_nl01[:, 1:].T
        oa_idx = np.where(n1_cos < 0.0)[0]

        psi = np.arcsin(unit_nl01[:, 2])
        if self.debug_level > 0:
            print('psi', psi)

        vl_dul = self.iL_vectors_dul
        nl0_dul0, nl1_dul1 = np.einsum('fi...->if...', self.iL_F_normals_du)
        if self.debug_level > 0:
            print(cp.iL_N.shape)
            print('vl_dul', vl_dul.shape)
            print(vl_dul)
            print('nl0_dul0', nl0_dul0.shape)
            print(nl0_dul0)
            print('nl1_dul1', nl1_dul1.shape)
            print(nl1_dul1)

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
            print('unit_nl0_dul0', unit_nl0_dul0.shape)
            print(unit_nl0_dul0)
            print('unit_nl1_dul1', unit_nl1_dul1.shape)
            print(unit_nl1_dul1)
            print('unit_vl_dul', unit_vl_dul.shape)
            print(unit_vl_dul)

        Tl0_dul0 = np.einsum('ij...->ji...',
                             np.array([np.zeros_like(unit_nl0_dul0),
                                       unit_nl0_dul0,
                                       np.einsum(
                                      '...j,...kNd,...ijk->...iNd',
                                      unit_vl, unit_nl0_dul0, EPS)
                             ]
                             ))

        if self.debug_level > 0:
            print('Tl0_dul0', Tl0_dul0.shape)
            print(Tl0_dul0)

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
            print('Tl0_dul0', Tl0_dul.shape)
            print(Tl0_dul)

        rho = 1 / np.sqrt((1 - unit_nl01[:, 2]**2))
        if self.debug_level > 0:
            print('rho', unit_nl01[:, 2])

        unit_nl01_dul = np.einsum(
            '...,...j,...ijNd->...iNd',
            rho, unit_nl1, Tl0_dul)[:, 2, ...]
        unit_nl01_dul0 = np.einsum(
            '...,...j,...ijNd->...iNd',
            rho, unit_nl1, Tl0_dul0)[:, 2, ...]
        unit_nl01_dul1 = np.einsum(
            '...,...jNd,...ij->...iNd',
            rho, unit_nl1_dul1, Tl0)[:, 2, ...]
        if self.debug_level > 0:
            print('unit_nl01_dul', unit_nl01_dul.shape)
            print(unit_nl01_dul)
            print('unit_nl01_dul0', unit_nl01_dul0.shape)
            print(unit_nl01_dul0)
            print('unit_nl01_dul1', unit_nl01_dul1.shape)
            print(unit_nl01_dul1)

        # get the map of facet nodes attached to interior lines
        iL0_N_map = self.F_N[self.iL_F[:, 0]].reshape(self.n_iL, -1)
        iL1_N_map = self.F_N[self.iL_F[:, 1]].reshape(self.n_iL, -1)
        iL_N_map = self.F_L_N[self.iL_within_F0]

        # enumerate the interior lines and broadcast it N and D into dimensions
        iL_map = np.arange(self.n_iL)[:, np.newaxis, np.newaxis]
        # broadcast the facet node map into D dimension
        l0_map = iL0_N_map[:, :, np.newaxis]
        l1_map = iL1_N_map[:, :, np.newaxis]
        l_map = iL_N_map[:, :, np.newaxis]
        # broadcast the spatial dimension map into iL and N dimensions
        D_map = np.arange(3)[np.newaxis, np.newaxis, :]
        # allocate the gamma derivatives of iL with respect to N and D
        # dimensions
        psi_du = np.zeros((self.n_iL, self.n_N, self.n_D), dtype='float_')
        # add the contributions gamma_du from the left and right facet
        # Note: this cannot be done in a single step since the incremental
        # assembly is not possible within a single index expression.
        psi_du[iL_map, l_map, D_map] += unit_nl01_dul
        if self.debug_level > 0:
            print('l_map', l_map.shape)
            print(l_map)
            print('psi_du', psi_du.shape)
            print(psi_du)
        psi_du[iL_map, l0_map, D_map] += unit_nl01_dul0
        if self.debug_level > 0:
            print('l0_map', l0_map.shape)
            print(l0_map)
            print('psi_du', psi_du.shape)
            print(psi_du)
        psi_du[iL_map, l1_map, D_map] += unit_nl01_dul1
        if self.debug_level > 0:
            print('l1_map', l1_map.shape)
            print(l1_map)
            print('psi_du', psi_du.shape)
            print(psi_du)
        psi_du[oa_idx, ...] *= -1.0
        return psi_du


class CreaseFacetOperators(HasStrictTraits):

    r'''Operators evaluating the instantaneous states of the facets.
    '''

    # =========================================================================
    # Property operators for initial configuration
    # =========================================================================
    F0_normals = Property(Array, depends_on='X, L, F')
    r'''Normal facet vectors.
    '''
    @cached_property
    def _get_F0_normals(self):
        x_F = self.x_0[self.F]
        N_deta_ip = self.Na_deta
        r_deta = np.einsum('ajK,IKi->Iaij', N_deta_ip, x_F)
        Fa_normals = np.einsum('Iai,Iaj,ijk->Iak',
                               r_deta[..., 0], r_deta[..., 1], EPS)
        return np.sum(Fa_normals, axis=1)

    sign_normals = Property(Array, depends_on='X,L,F')
    r'''Orientation of the normal in the initial state.
    This array is used to switch the normal vectors of the faces
    to be oriented in the positive sense of the z-axis.
    '''
    @cached_property
    def _get_sign_normals(self):
        return np.sign(self.F0_normals[:, 2])

    F_N = Property(Array, depends_on='X,L,F')
    r'''Counter-clockwise enumeration.
    '''
    @cached_property
    def _get_F_N(self):
        turn_facets = np.where(self.sign_normals < 0)
        F_N = np.copy(self.F)
        F_N[turn_facets, :] = self.F[turn_facets, ::-1]
        return F_N

    F_normals = Property(Array, depends_on=INPUT)
    r'''Get the normals of the facets.
    '''
    @cached_property
    def _get_F_normals(self):
        n = self.Fa_normals
        return np.sum(n, axis=1)

    F_normals_0 = Property(Array, depends_on=INPUT)
    r'''Get the normals of the facets.
    '''
    @cached_property
    def _get_F_normals_0(self):
        n = self.Fa_normals_0
        return np.sum(n, axis=1)

    norm_F_normals = Property(Array, depends_on=INPUT)
    r'''Get the normed normals of the facets.
    '''
    @cached_property
    def _get_norm_F_normals(self):
        n = self.F_normals
        mag_n = np.sqrt(np.einsum('...i,...i', n, n))
        return n / mag_n[:, np.newaxis]

    norm_F_normals_0 = Property(Array, depends_on=INPUT)
    r'''Get the normed normals of the facets.
    '''
    @cached_property
    def _get_norm_F_normals_0(self):
        n = self.F_normals_0
        mag_n = np.sqrt(np.einsum('...i,...i', n, n))
        return n / mag_n[:, np.newaxis]

    F_normals_du = Property(Array, depends_on=INPUT)
    r'''Get the normals of the facets.
    '''
    @cached_property
    def _get_F_normals_du(self):
        n_du = self.Fa_normals_du
        return np.sum(n_du, axis=1)

    F_area = Property(Array, depends_on=INPUT)
    r'''Get the surface area of the facets.
    '''
    @cached_property
    def _get_F_area(self):
        a = self.Fa_area
        A = np.einsum('a,Ia->I', self.eta_w, a)
        return A

    # =========================================================================
    # Potential energy
    # =========================================================================
    F_V = Property(Array, depends_on=INPUT)
    r'''Get the total potential energy of gravity for each facet
    '''
    @cached_property
    def _get_F_V(self):
        eta_w = self.eta_w
        a = self.Fa_area
        ra = self.Fa_r
        F_V = np.einsum('a,Ia,Ia->I', eta_w, ra[..., 2], a)
        return F_V

    F_V_du = Property(Array, depends_on=INPUT)
    r'''Get the derivative of total potential energy of gravity for each facet
    with respect to each node and displacement component [FIi]
    '''
    @cached_property
    def _get_F_V_du(self):
        r = self.Fa_r
        a = self.Fa_area
        a_dx = self.Fa_area_du
        r3_a_dx = np.einsum('Ia,IaJj->IaJj', r[..., 2], a_dx)
        N_eta_ip = self.Na
        r3_dx = np.einsum('aK,KJ,j->aJj', N_eta_ip, DELTA, DELTA[2, :])
        a_r3_dx = np.einsum('Ia,aJj->IaJj', a, r3_dx)
        F_V_du = np.einsum('a,IaJj->IJj', self.eta_w, (a_r3_dx + r3_a_dx))
        return F_V_du

    # =========================================================================
    # Line vectors
    # =========================================================================

    F_L_vectors_0 = Property(Array, depends_on=INPUT)
    r'''Get the cycled line vectors around the facet
    The cycle is closed - the first and last vector are identical.

    .. math::
        v_{pld} \;\mathrm{where} \; p\in\mathcal{F}, l\in (0,1,2), d\in (0,1,2)

    with the indices :math:`p,l,d` representing the facet, line vector around
    the facet and and vector component, respectively.
    '''
    @cached_property
    def _get_F_L_vectors_0(self):
        F_N = self.F_N  # F_N is cycled counter clockwise
        return self.x_0[F_N[:, (1, 2, 0)]] - self.x_0[F_N[:, (0, 1, 2)]]

    F_L_vectors = Property(Array, depends_on=INPUT)
    r'''Get the cycled line vectors around the facet
    The cycle is closed - the first and last vector are identical.

    .. math::
        v_{pld} \;\mathrm{where} \; p\in\mathcal{F}, l\in (0,1,2), d\in (0,1,2)

    with the indices :math:`p,l,d` representing the facet, line vector around
    the facet and and vector component, respectively.
    '''
    @cached_property
    def _get_F_L_vectors(self):
        F_N = self.F_N  # F_N is cycled counter clockwise
        return self.x[F_N[:, (1, 2, 0)]] - self.x[F_N[:, (0, 1, 2)]]

    F_L_vectors_du = Property(Array, depends_on=INPUT)
    r'''Get the derivatives of the line vectors around the facets.

    .. math::
        \pard{v_{pld}}{x_{Ie}} \; \mathrm{where} \;
        p \in \mathcal{F},  \in (0,1,2), d\in (0,1,2), I\in \mathcal{N},
        e \in (0,1,3)

    with the indices :math:`p,l,d,I,e` representing the facet,
    line vector around the facet and and vector component,
    node vector and and its component index,
    respectively.

    This array works essentially as an index function delivering -1
    for the components of the first node in each dimension and +1
    for the components of the second node
    in each dimension.

    For a facet :math:`p` with lines :math:`l` and component :math:`d` return
    the derivatives with respect to the displacement of the node :math:`I`
    in the direction :math:`e`.

    .. math::
        \bm{a}_1 = \bm{x}_2 - \bm{x}_1 \\
        \bm{a}_2 = \bm{x}_3 - \bm{x}_2 \\
        \bm{a}_3 = \bm{x}_1 - \bm{x}_3

    The corresponding derivatives are then

    .. math::
        \pard{\bm{a}_1}{\bm{u}_1} = -1, \;\;\;
        \pard{\bm{a}_1}{\bm{u}_2} = 1 \\
        \pard{\bm{a}_2}{\bm{u}_2} = -1, \;\;\;
        \pard{\bm{a}_2}{\bm{u}_3} = 1 \\
        \pard{\bm{a}_3}{\bm{u}_3} = -1, \;\;\;
        \pard{\bm{a}_3}{\bm{u}_1} = 1 \\

    '''

    def _get_F_L_vectors_du(self):
        return self.L_vectors_du[self.F_L]

    F_L_vectors_dul = Property(Array, depends_on=INPUT)

    def _get_F_L_vectors_dul(self):
        return self.L_vectors_dul[self.F_L]

    norm_F_L_vectors = Property(Array, depends_on=INPUT)
    r'''Get the cycled line vectors around the facet
    The cycle is closed - the first and last vector are identical.
    '''
    @cached_property
    def _get_norm_F_L_vectors(self):
        v = self.F_L_vectors
        mag_v = np.sqrt(np.einsum('...i,...i', v, v))
        return v / mag_v[..., np.newaxis]

    norm_F_L_vectors_du = Property(Array, depends_on=INPUT)
    '''Get the derivatives of cycled line vectors around the facet
    '''
    @cached_property
    def _get_norm_F_L_vectors_du(self):
        v = self.F_L_vectors
        v_du = self.F_L_vectors_du  # @UnusedVariable
        mag_v = np.einsum('...i,...i', v, v)  # @UnusedVariable
        # @todo: finish the chain rule
        raise NotImplemented

    # =========================================================================
    # Orthonormal basis of each facet.
    # =========================================================================
    F_L_bases = Property(Array, depends_on=INPUT)
    r'''Line bases around a facet.
    '''
    @cached_property
    def _get_F_L_bases(self):
        l = self.norm_F_L_vectors
        n = self.norm_F_normals
        lxn = np.einsum('...li,...j,...kij->...lk', l, n, EPS)
        n_ = n[:, np.newaxis, :] * np.ones((1, 3, 1), dtype='float_')
        T = np.concatenate([l[:, :, np.newaxis, :],
                            -lxn[:, :, np.newaxis, :],
                            n_[:, :, np.newaxis, :]], axis=2)
        return T

    F_L_bases_du = Property(Array, depends_on=INPUT)
    r'''Derivatives of the line bases around a facet.
    '''
    @cached_property
    def _get_F_L_bases_du(self):
        '''Derivatives of line bases'''
        raise NotImplemented

    # =========================================================================
    # Sector angles
    # =========================================================================
    F_theta = Property(Array, depends_on=INPUT)
    '''Get the sector angles :math:`\theta`  within a facet.
    '''
    @cached_property
    def _get_F_theta(self):
        v = self.F_L_vectors
        a = -v[:, (2, 0, 1), :]
        b = v[:, (0, 1, 2), :]
        return get_theta(a, b)

    F_theta_du = Property(Array, depends_on=INPUT)
    r'''Get the derivatives of sector angles :math:`\theta` within a facet.
    '''
    @cached_property
    def _get_F_theta_du(self):
        v = self.F_L_vectors
        v_du = self.F_L_vectors_du

        a = -v[:, (2, 0, 1), :]
        b = v[:, (0, 1, 2), :]
        a_du = -v_du[:, (2, 0, 1), ...]
        b_du = v_du[:, (0, 1, 2), ...]

        return get_theta_du(a, a_du, b, b_du)

    # =========================================================================
    # Surface integrals using numerical integration
    # =========================================================================
    eta_ip = Array('float_')
    r'''Integration points within a triangle.
    '''

    def _eta_ip_default(self):
        return np.array([[1. / 3., 1. / 3.]], dtype='f')

    eta_w = Array('float_')
    r'''Weight factors for numerical integration.
    '''

    def _eta_w_default(self):
        return np.array([1. / 2.], dtype='f')

    Na = Property(depends_on='eta_ip')
    r'''Shape function values in integration points.
    '''
    @cached_property
    def _get_Na(self):
        eta = self.eta_ip
        return np.array([eta[:, 0], eta[:, 1], 1 - eta[:, 0] - eta[:, 1]],
                        dtype='f').T

    Na_deta = Property(depends_on='eta_ip')
    r'''Derivatives of the shape functions in the integration points.
    '''
    @cached_property
    def _get_Na_deta(self):
        return np.array([[[1, 0, -1],
                          [0, 1, -1]],
                         ], dtype='f')

    Fa_normals_du = Property
    '''Get the derivatives of the normals with respect
    to the node displacements.
    '''

    def _get_Fa_normals_du(self):
        x_F = self.x[self.F_N]
        N_deta_ip = self.Na_deta
        NN_delta_eps_x1 = np.einsum('aK,aL,KJ,dli,ILl->IaiJd',
                                    N_deta_ip[:, 0, :], N_deta_ip[:, 1, :],
                                    DELTA, EPS, x_F)
        NN_delta_eps_x2 = np.einsum('aK,aL,LJ,kdi,IKk->IaiJd',
                                    N_deta_ip[:, 0, :], N_deta_ip[:, 1, :],
                                    DELTA, EPS, x_F)
        n_du = NN_delta_eps_x1 + NN_delta_eps_x2
        return n_du

    Fa_area_du = Property
    '''Get the derivatives of the facet area with respect
    to node displacements.
    '''

    def _get_Fa_area_du(self):
        a = self.Fa_area
        n = self.Fa_normals
        n_du = self.Fa_normals_du
        a_du = np.einsum('Ia,Iak,IakJd->IaJd', 1 / a, n, n_du)
        return a_du

    Fa_normals = Property
    '''Get normals of the facets.
    '''

    def _get_Fa_normals(self):
        x_F = self.x[self.F_N]
        N_deta_ip = self.Na_deta
        r_deta = np.einsum('ajK,IKi->Iaij', N_deta_ip, x_F)
        return np.einsum('Iai,Iaj,ijk->Iak',
                         r_deta[..., 0], r_deta[..., 1], EPS)

    Fa_normals_0 = Property
    '''Get normals of the facets.
    '''

    def _get_Fa_normals_0(self):
        x_F = self.x_0[self.F_N]
        N_deta_ip = self.Na_deta
        r_deta = np.einsum('ajK,IKi->Iaij', N_deta_ip, x_F)
        return np.einsum('Iai,Iaj,ijk->Iak',
                         r_deta[..., 0], r_deta[..., 1], EPS)

    Fa_area = Property
    '''Get the surface area of the facets.
    '''

    def _get_Fa_area(self):
        n = self.Fa_normals
        a = np.sqrt(np.einsum('Iai,Iai->Ia', n, n))
        return a

    Fa_r = Property
    '''Get the reference vector to integrations point in each facet.
    '''

    def _get_Fa_r(self):
        x_F = self.x[self.F_N]
        N_eta_ip = self.Na
        r = np.einsum('aK,IKi->Iai', N_eta_ip, x_F)
        return r

    # =========================================================================
    # Interier as level set
    # =========================================================================


class CreaseCummulativeOperators(HasStrictTraits):

    '''Characteristics of the whole crease pattern.
    '''
    V = Property(Float, depends_on=INPUT)
    '''Get the total potential energy of gravity
    '''
    @cached_property
    def _get_V(self):
        return np.sum(self.F_V)

    V_du = Property(Array, depends_on=INPUT)
    '''Get the gradient of potential energy with respect
    to the current nodal position.
    '''
    @cached_property
    def _get_V_du(self):
        F = self.F_N
        F_V_du = self.F_V_du
        dof_map = (3 * F[:, :, np.newaxis] +
                   np.arange(3)[np.newaxis, np.newaxis, :])
        V_du = np.bincount(dof_map.flatten(), weights=F_V_du.flatten())
        return V_du


class CreaseViewRelatedOperators(HasStrictTraits):

    '''Characteristics of the whole crease pattern.
    '''
    center = Property(Array, depends_on=INPUT)
    '''Get the total potential energy of gravity
    '''
    @cached_property
    def _get_center(self):
        return np.sum(self.bounding_box, axis=0) / 2.0

    bounding_box = Property(Array, depends_on=INPUT)
    '''Get the gradient of potential energy with respect
    to the current nodal position.
    '''
    @cached_property
    def _get_bounding_box(self):
        x_min, x_max = np.min(self.x, axis=0), np.max(self.x, axis=0)
        return np.array([x_min, x_max])


if __name__ == '__main__':
    from .crease_pattern_state import CreasePatternState
    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [1, 1, 0],
                               [0, 1, 0],
                               [3, 2, 1]],
                            L=[[0, 1], [1, 2], [3, 2], [0, 3],
                               [0, 2], [1, 4], [2, 4], [3, 4]],
                            F=[[0, 1, 2], [2, 0, 3], [1, 2, 4], [2, 3, 4]])

    # structural mappings

    print('nodes')
    print(cp.X)
    print('lines')
    print(cp.L)
    print('faces')
    print(cp.F)
    print('faces counter-clockwise')
    print(cp.F_N)
    print('node neighbors')
    print(cp.N_nbr)
    print('interior nodes')
    print(cp.iN)
    print('edge nodes')
    print(cp.eN)
    print('interior node neighbor (cycles ordered)')
    print(cp.iN_nbr)
    print('lines associated with the interior nodes')
    print(cp.iN_aln)
    print('interior lines')
    print(cp.iL)
    print('edge lines')
    print(cp.eL)
    print('iL_F: faces of interior lines')
    print(cp.iL_F)
    print('F_L: lines of faces')
    print(cp.F_L)

    # dependent attributes in initial state

    print('L_lengths: line lengths')
    print(cp.L_lengths)
    print()
    print('F0_normals: facet normals')
    print(cp.F0_normals)
    print()
    print('F_area: facet area')
    print(cp.F_area)
    print()
    print('iN_theta: angles around the interior nodes')
    print(cp.iN_theta)
    print()
    print('iL_psi: dihedral angles around interior lines')
    print(cp.iL_psi)
    print()

    u = np.zeros_like(cp.x_0)
    cp.u = u
    # dependent attributes in initial state

    print('L_lengths: line lengths')
    print(cp.L_lengths)
    print()
    print('F_normals: facet normals')
    print(cp.F_normals)
    print()
    print('F_area: facet area')
    print(cp.F_area)
    print()
    print('iN_theta: angles around the interior nodes')
    print(cp.iN_theta)
    print()
    print('iL_psi: dihedral angles around interior lines')
    print(cp.iL_psi)
    print()
#     print 'iL_psi_du: dihedral angles around interior lines'
#     print cp.get_iL_psi_du(u)
#     print
    print('F_theta: crease angles within each triangular facet')
    print(cp.F_theta)
    print()
    print('NN_theta: ')
    print(cp.NN_theta)

    # tests the counter-clockwise enumeration of facets (the second faces
    # is reversed from [2,0,3] to [3,0,2]
    # test the face angles around a node
    print('F_N')
    print(cp.F_N)
    print()
    print('iN_L')
    print(cp.iN_aln)
    print()
    print('F_theta')
    print(cp.F_theta)
    print()
    print('NN_theta')
    print(cp.NN_theta)
    print()
    print('iN')
    print(cp.iN)
    print()
    print('iN_neighbors')
    print(cp.iN_nbr)
    print()
    print('iN_theta')
    print(cp.iN_theta)
    print()
    print('F_L_vectors')
    print(cp.F_L_vectors)
    print()
    print('L_vectors_du')
    print(cp.L_vectors_du)
    print()
    print('F_L_vectors_du')
    print(cp.F_L_vectors_du)
    print()
    print('F_theta')
    print([cp.F_theta])
    print()
    print('F_theta_du')
    print([cp.F_theta_du])
    print()
    print('NN_theta_du')
    print(cp.NN_theta_du)
    print()
    print('iN_theta_du')
    print(cp.iN_theta_du)
    print()

    print('----------------')
    print('  VIEW RELATED  ')
    print('----------------')
    print('bounding box')
    print(cp.bounding_box)
    print('center')
    print(cp.center)
