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


import types

from traits.api import \
    implements,  List, Tuple, Float, \
    Property, Array, Int

from fu import \
    Fu
from fu_poteng_bending_viz3d import \
    FuPotEngBendingViz3D
from fu_poteng_node_load_viz3d import \
    FuPotEngNodeLoadViz3D
import numpy as np
from oricreate.crease_pattern.crease_pattern_operators import CreaseCummulativeOperators
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

    exclude_lines = Array(int, value=[])

    rho = Float(0.236, label='material density',
                enter_set=True, auto_set=False)

    thickness = Float(0.01, label='thickness',
                      enter_set=True, auto_set=False)

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

    def _get_F_ext(self, t=0):
        cp = self.forming_task.formed_object
        F_ext = np.zeros_like(cp.u, dtype='float_')
        for node, dim, value in self.F_ext_list:
            if isinstance(value, types.FunctionType):
                F_ext[node, dim] = value(t)
            else:
                F_ext[node, dim] = value
        return F_ext

    fu_factor = Float(1.0)

    def get_f(self, t=0):
        '''Get the total potential energy.
        '''
        cp = self.forming_task.formed_object

        iL_mask = np.ones_like(cp.iL, dtype=bool)
        iL_mask[cp.L_iL[self.exclude_lines]] = True

        iL_phi = cp.iL_psi[iL_mask] - cp.iL_psi_0[iL_mask]
        iL_length = np.linalg.norm(cp.iL_vectors[iL_mask], axis=1)

        stored_energy = np.einsum(
            '...i,...i,...i->...', self.kappa, iL_phi**2, iL_length) / 2.0

#         F_ext = self._get_F_ext(t)
#         ext_energy = np.einsum(
#             '...i,...i->...', F_ext.flatten(), cp.u.flatten())
#         tot_energy = self.fu_factor * (stored_energy - ext_energy)
#         return tot_energy

        F_ext = self._get_F_ext(t)
        V_ext = cp.V * self.rho * self.thickness
        ext_energy = (np.einsum(
            '...i,...i->...', F_ext.flatten(), cp.u.flatten()) - V_ext)
        tot_energy = self.fu_factor * (stored_energy - ext_energy)
        return tot_energy

    def get_f_du(self, t=0):
        '''Get the derivatives with respect to individual displacements.
        '''
        cp = self.forming_task.formed_object

        iL_mask = np.ones_like(cp.iL, dtype=bool)
        iL_mask[cp.L_iL[self.exclude_lines]] = True

        F_ext = self._get_F_ext(t)

        iL_phi = cp.iL_psi[iL_mask] - cp.iL_psi_0[iL_mask]
        iL_phi_du = cp.iL_psi_du[iL_mask]
        iL_length = np.linalg.norm(cp.iL_vectors[iL_mask], axis=1)

        Pi_int_du = np.einsum('...l,...l,...l,...lId->...Id',
                              iL_length, self.kappa, iL_phi, iL_phi_du)

        V_du = cp.V_du.reshape((-1, 3))

        Pi_ext_du = F_ext - V_du * self.rho * self.thickness

        Pi_du = Pi_int_du - Pi_ext_du

        return Pi_du.flatten()

    viz3d_classes = dict(default=FuPotEngBendingViz3D,
                         node_load=FuPotEngNodeLoadViz3D)
