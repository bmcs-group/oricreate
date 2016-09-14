import numpy as np
from oricreate.api import CreasePatternState, CustomCPFactory
from oricreate.api import GuConstantLength, GuDofConstraints, \
    SimulationConfig, SimulationTask, fix, FTV
from oricreate.fu import \
    FuPotEngTotal


def create_cp_factory():
    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [1, 1, 0],
                               [0, 1, 0],
                               ],
                            L=[[0, 1],
                               [1, 2],
                               [2, 0],
                               [2, 3],
                               [3, 0],
                               ],
                            F=[[0, 2, 1], [0, 2, 3],
                               ]
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    return cp_factory

if __name__ == '__main__':

    cp_factory_task = create_cp_factory()

    cp = cp_factory_task.formed_object

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()

    dof_constraints = fix([1], [0, 1, 2]) + fix([0], [2]) + fix([2], [0, 2])

    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

    sim_config = SimulationConfig(goal_function_type='total potential energy',
                                  debug_level=0,
                                  use_f_du=True,
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints},
                                  acc=1e-7, MAX_ITER=1000)

    FN = lambda F: lambda t: t * F
    P = 1
    F_ext_list = [(3, 2, FN(-P))]

    fu_tot_poteng = FuPotEngTotal(kappa=np.array([100]), fu_factor=1,
                                  F_ext_list=F_ext_list)

    sim_config._fu = fu_tot_poteng
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=10)

    cp = sim_task.formed_object
    cp.x_0 = np.copy(sim_task.previous_task.formed_object.X)
    cp.u[:, :] = 0.0
    fu_tot_poteng.forming_task = sim_task

    cp.u[1, 2] = -0.0001
    cp.u[3, 2] = -0.0001
    sim_task.u_1

    cp = sim_task.formed_object
    iL_phi = cp.iL_psi2 - cp.iL_psi_0
    print 'phi',  iL_phi

    print 'V', cp.V
    print 'F_V', cp.F_V
    V_du = cp.V_du.reshape((-1, 3))
    print 'V_du', V_du

    iL_length = np.linalg.norm(cp.iL_vectors, axis=1)
    iL_m = sim_config._fu.kappa * iL_phi * iL_length

    print 'moments', iL_m

    ftv = FTV()

    ftv.add(sim_task.formed_object.viz3d)
    ftv.add(gu_dof_constraints.viz3d)
    ftv.add(fu_tot_poteng.viz3d)
    ftv.add(fu_tot_poteng.viz3d_dict['node_load'])
    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()
