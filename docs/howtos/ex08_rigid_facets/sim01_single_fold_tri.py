r'''

Fold the logo of Engineered Folding Research Center
---------------------------------------------------

This example shows one possible  

The target face is defined as horizontal plane at the height 8
and nodes [0,1,2] are involved in the minimum distance criterion.
'''

from oricreate.gu import GuConstantLength, GuDofConstraints, fix
from oricreate.simulation_step import \
    SimulationStep, SimulationConfig


def create_cp_factory():
    # begin
    import numpy as np
    from oricreate.api import CreasePatternState, CustomCPFactory

    x = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [2, 1, 0]
                  ], dtype='float_')

    L = np.array([[0, 1], [1, 2], [2, 0],
                  [1, 3], [2, 3]],
                 dtype='int_')

    F = np.array([[0, 1, 2],
                  [1, 3, 2],
                  ], dtype='int_')

    cp = CreasePatternState(X=x,
                            L=L,
                            F=F
                            )
    cp.u[3, 2] = 0.1

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory

if __name__ == '__main__':

    import mayavi.mlab as m

    cp_factory = create_cp_factory()
    cp = cp_factory.formed_object

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()
    dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) + fix([2], [2]) + \
        fix([3], 2, 0.2)
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

    sim_config = SimulationConfig(gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints})
    sim_step = SimulationStep(forming_task=cp_factory,
                              config=sim_config, acc=1e-5, MAX_ITER=10)

    sim_step._solve_nr(1.0)

    m.figure(bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.6, 0.6, 0.6))
    cp.plot_mlab(m, nodes=True, lines=True)

    m.show()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=True)
    plt.tight_layout()
    plt.show()
