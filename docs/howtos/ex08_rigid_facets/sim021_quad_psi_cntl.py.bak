r'''

Fold the logo of Engineered Folding Research Center
---------------------------------------------------

This example shows one possible  

The target face is defined as horizontal plane at the height 8
and nodes [0,1,2] are involved in the minimum distance criterion.
'''

import numpy as np
from oricreate.gu import \
    GuConstantLength, GuDofConstraints, GuPsiConstraints, fix
from oricreate.simulation_step import \
    SimulationStep, SimulationConfig


def create_cp_factory():
    # begin
    from oricreate.api import CreasePatternState, CustomCPFactory

    x = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [2, 0, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [2, 1, 0],
                  ], dtype='float_')

    L = np.array([[0, 1], [1, 2],
                  [3, 4], [4, 5],
                  [0, 3], [1, 4], [2, 5],
                  [0, 4], [1, 5]],
                 dtype='int_')

    F = np.array([[0, 1, 4],
                  [1, 2, 5],
                  [0, 4, 3],
                  [1, 5, 4],
                  ], dtype='int_')

    cp = CreasePatternState(X=x,
                            L=L,
                            F=F
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory


if __name__ == '__main__':

    import mayavi.mlab as m
    import pylab as p
    cp_factory_task = create_cp_factory()
    cp = cp_factory_task.formed_object
    ax = p.axes()
    cp.plot_mpl(ax)
    p.show()
    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()
    dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) + fix([4], [2])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
    psi_max = np.pi / 4.0
    gu_psi_constraints = \
        GuPsiConstraints(forming_task=cp_factory_task,
                         psi_constraints=[([(5, 1.0)], lambda t: psi_max * t),
                                          ([(7, 1.0)], 0),
                                          ([(8, 1.0)], 0),
                                          ])

    sim_config = SimulationConfig(gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints,
                                      'psi': gu_psi_constraints},
                                  acc=1e-5, MAX_ITER=10)
    sim_step = SimulationStep(forming_task=cp_factory_task,
                              config=sim_config)

    sim_step._solve_nr()

    m.figure(bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.6, 0.6, 0.6))
    cp.plot_mlab(m, nodes=True, lines=True)

    m.show()
