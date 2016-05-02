'''
'''
from traits.api import implements

from custom_factory_mpl import create_cp_factory
import numpy as np
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix, FTV
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
        F_ext[3, 2] = -4.0
        ext_energy = np.einsum(
            '...i,...i->...', F_ext.flatten(), cp.u.flatten())
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
                                  debug_level=0,
                                  use_f_du=False,
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints},
                                  acc=1e-5, MAX_ITER=100)
    sim_config._fu = FuStoredEnergy()
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=1)

    cp.u[3, 2] = 0.001
    print 'kinematic simulation: u', sim_task.u_1

    ftv = FTV()
    ftv.add(sim_task.formed_object.viz3d)
    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()
