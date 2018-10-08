import math
from traits.api import \
    HasStrictTraits, Instance, Property, cached_property, \
    Array, Int

import matplotlib.pyplot as \
    plt
import numpy as np
from oricreate.api import \
    IFormingTask, SimulationHistory, \
    SimulationStep, SimulationConfig
from oricreate.gu import \
    GuConstantLength, GuDofConstraints, fix
from oricreate.simulation_tasks.simulation_task import SimulationTask
r'''

Construct a crease pattern for twist folding and simulate it 
------------------------------------------------------------

This example shows quadrilateral facets with a twist fold.

Modeling
========
@todo: introduce folding time
@todo: separate the facets from triangulation - include an example in crease pattern factory.
@todo: automated constraint generation to preserve planarity of facets for polygonal facet shapes 
       - using either constant length constraint or parallel normals
@todo: check folding angles
@todo: check folding angle derivatives 

'''


def damped_range(low, high, n):
    phi_range = np.linspace(0, np.pi, n)
    u_range = (1 - np.cos(phi_range)) / 2.0
    return low + (high - low) * u_range


def create_cp_factory():
    # begin
    from oricreate.api import CreasePatternState, CustomCPFactory

    x = np.array([[0, 0, 0],
                  [2, 0, 0],
                  [3, 0, 0],
                  [4, 0, 0],
                  [0, 1, 0],
                  [2, 1, 0],
                  [0, 2, 0],
                  [1, 2, 0],
                  [3, 2, 0],
                  [4, 2, 0],
                  [2, 3, 0],
                  [4, 3, 0],
                  [0, 4, 0],
                  [1, 4, 0],
                  [2, 4, 0],
                  [4, 4, 0],
                  ], dtype='float_')

    L = np.array([[0, 1], [1, 2], [2, 3],
                  [4, 5],
                  [6, 7],
                  [8, 9],
                  [10, 11],
                  [12, 13], [13, 14],  [14, 15],
                  [4, 6],
                  [0, 4],
                  [6, 12],
                  [7, 13],
                  [1, 5],  [10, 14],
                  [2, 8],
                  [3, 9],
                  [9, 11],  [11, 15],
                  [7, 5],
                  [5, 8],
                  [8, 10], [10, 7]
                  ],
                 dtype='int_')

    F = np.array([[0, 1, 5, 4],
                  [1, 2, 8, 5],
                  [3, 9, 8, 2],
                  [9, 11, 10, 8],
                  [15, 14, 10, 11],
                  [14, 13, 7, 10],
                  [12, 6, 7, 13],
                  [6, 4, 5, 7],
                  [7, 5, 8, 10]
                  ], dtype='int_')

    L_range = np.arange(len(L))

    x_mid = (x[F[:, 1]] + x[F[:, 3]]) / 2.0
    x_mid[:, 2] -= 0.5
    n_F = len(F)
    n_x = len(x)
    x_mid_i = np.arange(n_x, n_x + n_F)

    L_mid = np.array([[F[:, 0], x_mid_i[:]],
                      [F[:, 1], x_mid_i[:]],
                      [F[:, 2], x_mid_i[:]],
                      [F[:, 3], x_mid_i[:]]])
    L_mid = np.vstack([L_mid[0].T, L_mid[1].T, L_mid[2].T, L_mid[3].T])

    x_derived = np.vstack([x, x_mid])
    L_derived = np.vstack([L, F[:-1, (1, 3)], L_mid])
    F_derived = np.vstack([F[:, (0, 1, 3)], F[:, (1, 2, 3)]])

    cp = CreasePatternState(X=x_derived,
                            L=L_derived,
                            F=F_derived
                            )

    cp.viz3d.L_selection = L_range

    cp.u[(2, 3, 8, 9), 2] = 0.01
    cp.u[(6, 7, 12, 13), 2] = -0.005
    cp.u[(10, 11, 14, 15), 2] = 0.005
    print 'n_N', cp.n_N
    print 'n_L', cp.n_L
    print 'n_free', cp.n_dofs - cp.n_L

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory


class TwistFolding(SimulationTask):

    def _previous_task_default(self):
        return create_cp_factory()

    n_steps = Int(3)

    config = Property(Instance(SimulationConfig))

    @cached_property
    def _get_config(self):
        # Configure simulation
        gu_constant_length = GuConstantLength()
        dof_constraints = fix(
            [0], [0, 1, 2]) + fix([1], [1, 2]) + fix([5], [2]) + \
            fix([3], [0], lambda t: math.cos(math.pi * t) - 1)
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        return SimulationConfig(goal_function_type='none',
                                gu={'cl': gu_constant_length,
                                    'dofs': gu_dof_constraints},
                                acc=1e-5, MAX_ITER=1000)

    def plot2d(self):
        fig, ax = plt.subplots()
        self.cp.plot_mpl(ax, facets=True)
        plt.tight_layout()
        plt.show()
        return


if __name__ == '__main__':

    twist_folding = TwistFolding(n_steps=40)
    print twist_folding.u_t
    print twist_folding.u_1
    # print twist_folding.sim_history.u_t[-1]
