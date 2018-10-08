r'''

Fold control by dihedral multiple angles
----------------------------------------

Sequence of triangles connected by hinge lines is folded
by controlling the dihedral angle between the triangles.
'''

import numpy as np
from oricreate.api import \
    SimulationTask, SimulationConfig, \
    GuConstantLength, GuDofConstraints, GuPsiConstraints, fix, \
    FTV, FTA


def create_cp_factory():
    # begin
    from oricreate.api import CreasePatternState, CustomCPFactory

    x = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [2, 1, 0],
                  [2, 0, 0],
                  [1.5, -0.707, 0]
                  ], dtype='float_')

    L = np.array([[0, 1], [1, 2], [2, 0],
                  [1, 3], [2, 3], [1, 4], [3, 4], [1, 5], [4, 5]],
                 dtype='int_')

    F = np.array([[0, 1, 2],
                  [1, 3, 2],
                  [1, 4, 3],
                  [1, 4, 5]
                  ], dtype='int_')

    cp = CreasePatternState(X=x,
                            L=L,
                            F=F
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory


if __name__ == '__main__':

    cp_factory_task = create_cp_factory()
    cp = cp_factory_task.formed_object

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()

    psi_max = np.pi * 0.3
    gu_psi_constraints = \
        GuPsiConstraints(forming_task=cp_factory_task,
                         psi_constraints=[([(1, 1.0)], lambda t: psi_max * t),
                                          ([(3, 1.0)], psi_max),
                                          ([(5, 1.0)], lambda t: psi_max * t),
                                          ])

    dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) \
        + fix([2], [2])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

    sim_config = SimulationConfig(goal_function_type='none',
                                  gu={'cl': gu_constant_length,
                                      'u': gu_dof_constraints,
                                      'psi': gu_psi_constraints},
                                  acc=1e-5, MAX_ITER=100)
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=5)

    sim_task.u_1
    cp = sim_task.formed_object

    ftv = FTV()
    ftv.add(sim_task.sim_history.viz3d['node_numbers'], order=5)
    ftv.add(sim_task.sim_history.viz3d['cp'])
    ftv.add(gu_dof_constraints.viz3d['default'])

    fta = FTA(ftv=ftv)
    fta.init_view(a=200, e=35, d=5, f=(0, 0, 0), r=0)

    fta.plot()
    fta.configure_traits()
