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
    cached_property, Property, Array

from fu import \
    Fu
from fu_bending_energy_viz3d import \
    FuBendingEnergyViz3D
import numpy as np
from oricreate.opt import \
    IFu
from oricreate.viz3d import \
    Visual3D


class FuTotalPotentialEnergy(Fu, Visual3D):

    '''Optimization criteria based on minimum Bending energy of gravity.

    This plug-in class lets the crease pattern operators evaluate the
    integral over the spatial domain in an instantaneous configuration
    '''

    implements(IFu)

    F_ext_list = List(Tuple, [])

    _kappa_arr = Array(float, value=[])
    _kappa = Float(1.0)

    kappa = Property()

    def _set_kappa(self, value):
        if isinstance(value, np.ndarray):
            self._kappa_arr = value
        elif isinstance(value, float):
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
        return self.forming_task.formed_object.V_du

    viz3d_dict = Property

    @cached_property
    def _get_viz3d_dict(self):
        return dict(default=FuBendingEnergyViz3D(vis3d=self))
