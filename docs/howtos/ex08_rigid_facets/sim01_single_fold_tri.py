r'''

Fold the logo of Engineered Folding Research Center
---------------------------------------------------

This example shows one possible  

The target face is defined as horizontal plane at the height 8
and nodes [0,1,2] are involved in the minimum distance criterion.
'''

import numpy as np
from oricreate.api import \
    SimulationTask, SimulationConfig, \
    GuConstantLength, GuDofConstraints, fix, \
    FTV, FTA


def create_cp_factory():
    # begin
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

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory

if __name__ == '__main__':

    import mayavi.mlab as m

    cp_factory_task = create_cp_factory()
    cp = cp_factory_task.formed_object

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()

    u_max = 0.8
    displ_cntl = fix([3], 2, lambda t: u_max * t)
    dof_constraints = fix([0], [0, 1, 2]) + fix([1], [1, 2]) \
        + fix([2], [2]) + displ_cntl
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

    sim_config = SimulationConfig(goal_function_type='none',
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints},
                                  acc=1e-5, MAX_ITER=10)
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=5)

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
#
#     m.figure(bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.6, 0.6, 0.6))
#     cp.plot_mlab(m, nodes=True, lines=True)
#     m.show()
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     cp.plot_mpl(ax, facets=True)
#     plt.tight_layout()
#     plt.show()
