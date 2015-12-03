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
                  [2, 0, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [2, 1, 0],
                  ], dtype='float_')

    L = np.array([[0, 1], [1, 2],
                  [3, 4], [4, 5],
                  [0, 3], [1, 4], [2, 5]],
                 dtype='int_')

    F = np.array([[0, 1, 4, 3],
                  [1, 2, 5, 4],
                  ], dtype='int_')

    x_mid = (x[F[:, 1]] + x[F[:, 3]]) / 2.0
    x_mid[:, 2] -= 1.0
    n_F = len(F)
    n_x = len(x)
    x_mid_i = np.arange(n_x, n_x + n_F)

    L_mid = np.array([[F[:, 0], x_mid_i[:]],
                      [F[:, 1], x_mid_i[:]],
                      [F[:, 2], x_mid_i[:]],
                      [F[:, 3], x_mid_i[:]]])
    L_mid = np.vstack([L_mid[0].T, L_mid[1].T, L_mid[2].T, L_mid[3].T])

    x_derived = np.vstack([x, x_mid])
    L_derived = np.vstack([L, F[:, (1, 3)], L_mid])
    F_derived = np.vstack([F[:, (0, 1, 2)], F[:, (0, 2, 3)]])

    cp = CreasePatternState(X=x_derived,
                            L=L_derived,
                            F=F_derived
                            )

    cp.u[5, 2] = 0.01

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory

if __name__ == '__main__':

    import mayavi.mlab as m

    cp_factory = create_cp_factory()
    cp = cp_factory.formed_object

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()
    dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) + fix([3], [2]) + \
        fix([5], 2, 0.5)
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
