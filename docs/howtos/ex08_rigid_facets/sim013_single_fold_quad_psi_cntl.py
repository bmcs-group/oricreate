r'''

Fold control using dihedral angle with quadrilateral facets
-----------------------------------------------------------

This example shows the folding process controlled by a dihedral
angle between two facets. In addition to the previous
example, this one introduces quadrilateral facets that 
are symmetrically composed of triangles with fixed fold line - i.e.
dihedral angle ''psi'' is equal to zero. The example  represents 
a standard Miura-Ori vertex with all associated kinematic constraints.  
'''

import numpy as np
from oricreate.api import \
    SimulationTask, SimulationConfig, \
    GuConstantLength, GuDofConstraints, GuPsiConstraints, fix, \
    FTV, FTA


def create_cp_factory():
    # begin
    from oricreate.api import CreasePatternState, CustomCPFactory

    x = np.array([[-1, 0, 0],
                  [0, 0, 0],
                  [1, 1, 0],
                  [2, 0, 0],
                  [1, -1, 0],
                  [-1, 1, 0],
                  [-1, -1, 0],
                  [2, 1, 0],
                  [2, -1, 0],
                  ], dtype='float_')

    L = np.array([[0, 1], [1, 2], [1, 5],
                  [1, 3], [2, 3],
                  [1, 4], [3, 4],
                  #[1, 5],
                  [6, 1],
                  [0, 5], [2, 5],
                  [0, 6], [4, 6],
                  [3, 7], [2, 7],
                  [3, 8], [4, 8]
                  ],
                 dtype='int_')

    F = np.array([[5, 1, 2],
                  [1, 3, 2],
                  [1, 4, 3],
                  [1, 4, 6],
                  [0, 1, 5],
                  [0, 1, 6],
                  [3, 2, 7],
                  [3, 4, 8],
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

    # Link the crease factory with the constraint client
    gu_constant_length = GuConstantLength()

    psi_max = np.pi * .49
    gu_psi_constraints = \
        GuPsiConstraints(forming_task=cp_factory_task,
                         psi_constraints=[([(2, 1.0)], 0.0),
                                          ([(7, 1.0)], 0.0),
                                          ([(4, 1.0)], 0.0),
                                          ([(6, 1.0)], 0.0),
                                          ([(3, 1.0)], lambda t: -psi_max * t),
                                          #([(5, 1.0)], lambda t: psi_max * t),
                                          ])

    dof_constraints = fix([0], [1]) + fix([1], [0, 1, 2]) \
        + fix([2, 4], [2])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

    sim_config = SimulationConfig(goal_function_type='none',
                                  gu={'cl': gu_constant_length,
                                      'u': gu_dof_constraints,
                                      'psi': gu_psi_constraints},
                                  acc=1e-8, MAX_ITER=100)
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=25)
    cp.u[(0, 3), 2] = -0.1
    cp.u[(1), 2] = 0.1
    sim_task.u_1
    cp = sim_task.formed_object

    ftv = FTV()
    ftv.add(sim_task.sim_history.viz3d_dict['node_numbers'], order=5)
    ftv.add(sim_task.sim_history.viz3d)
    ftv.add(gu_dof_constraints.viz3d)

    fta = FTA(ftv=ftv)
    fta.init_view(a=200, e=35, d=5, f=(0, 0, 0), r=0)
    fta.add_cam_move(a=200, e=34, n=5, d=5, r=0,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(0, 1, 4),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.render()
    fta.configure_traits()
