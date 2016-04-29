'''
'''
from traits.api import implements, Bool

from custom_factory_mpl import create_cp_factory
import numpy as np
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix
from oricreate.fu.fu import \
    Fu
from oricreate.opt import \
    IFu


class FuStoredEnergy(Fu):

    '''Optimization criteria based on minimum potential energy of gravity.

    This plug-in class lets the crease pattern operators evaluate the
    integral over the spatial domain in an instantaneous configuration
    '''

    implements(IFu)

    def get_f(self, t=0):
        '''Get the potential energy of gravity.
        '''
        cp = self.forming_task.formed_object
        phi_iL = cp.iL_psi2 - cp.iL_psi_0
        stored_energy = np.einsum(
            '...i,...i->...', phi_iL**2, cp.L_lengths) / 2.0
        F_ext = np.zeros_like(cp.u, dtype='float_')
        F_ext[3, 2] = -10.0
        ext_energy = np.einsum(
            '...i,...i->...', F_ext.flatten(), cp.u.flatten() / 2.0)
        tot_energy = stored_energy - ext_energy
        print 'tot_energy', tot_energy
        return tot_energy

    def get_f_du(self, t=0):
        '''Get the derivatives with respect to individual displacements.
        '''
        return self.forming_task.formed_object.V_du

if __name__ == '__main__':
    cp_factory_task = create_cp_factory()
    cp = cp_factory_task.formed_object
    print 'x', cp.x

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()

    dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) \
        + fix([2], [2])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

    sim_config = SimulationConfig(goal_function_type='potential_energy',
                                  use_f_du=False,
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints},
                                  acc=1e-5, MAX_ITER=10)
    sim_config._fu = FuStoredEnergy()
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=1)

    cp.u[3, 2] = 0.01

    print 'kinematic simulation: u', sim_task.u_1

    print 'normed normal vectors of the facets\n', cp.norm_F_normals
    print 'normed normal facet vectors adjacent to the lines\n', cp.norm_iL_F_normals

    cp.u[3, 2] = 0.5
    cp.u = cp.u
    print 'iL_psi - u = 1.0', cp.iL_psi, cp.iL_psi2, cp.iL_psi_0
    phi_iL = cp.iL_psi2 - cp.iL_psi_0
    print 'iL_phi', phi_iL

    print 'normed normal vectors of the facets\n', cp.norm_F_normals
    print 'normed normal facet vectors adjacent to the lines\n', cp.norm_iL_F_normals

    # change the position
    cp.u[3, 2] = -0.5
    cp.u = cp.u
    print 'iL_psi - u = -1.0', cp.iL_psi, cp.iL_psi2, cp.iL_psi_0
    phi_iL = cp.iL_psi2 - cp.iL_psi_0
    print 'iL_phi', phi_iL

    stored_energy = np.einsum('...i,...i->...', phi_iL**2, cp.L_lengths) / 2.0
    print 'stored energy', stored_energy

    F_ext = np.zeros_like(cp.u, dtype='float_')
    F_ext[3, 2] = -1.0

    print 'normed normal vectors of the facets\n', cp.norm_F_normals
    print 'normed normal facet vectors adjacent to the lines\n', cp.norm_iL_F_normals

    import sympy as sp
    psi_, psi0_ = sp.symbols('psi,psi0')

    print sp.simplify(sp.asin(psi_) - sp.asin(psi0_))
    print sp.diff(sp.asin(psi_), psi_)
