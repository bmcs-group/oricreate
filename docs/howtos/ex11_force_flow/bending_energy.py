'''
'''
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix, FTV
from oricreate.fu import FuPotEngTotal

from custom_factory_mpl import create_cp_factory
import numpy as np


if __name__ == '__main__':
    cp_factory_task = create_cp_factory()
    cp = cp_factory_task.formed_object

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
    fu_tot_poteng = FuPotEngTotal(kappa=10,
                                  F_ext_list=[(3, 2, -1), (4, 2, -1)])
    sim_config._fu = fu_tot_poteng
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=1)

    cp.u[4, 2] = 0.001
    print('kinematic simulation: u', sim_task.u_1)

    cp = sim_task.formed_object
    iL_phi = cp.iL_psi2 - cp.iL_psi_0
    print('phi',  iL_phi)
    iL_length = np.linalg.norm(cp.iL_vectors, axis=1)
    iL_m = sim_config._fu.kappa * iL_phi * iL_length

    print('moments', iL_m)

    ftv = FTV()
    ftv.add(sim_task.formed_object.viz3d)
    ftv.add(gu_dof_constraints.viz3d)
    ftv.add(fu_tot_poteng.viz3d)
    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()
