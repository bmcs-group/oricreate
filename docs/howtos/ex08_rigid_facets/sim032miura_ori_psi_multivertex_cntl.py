r'''

Fold the Miura ori crease pattern using psi control
---------------------------------------------------

'''

import numpy as np
from oricreate.api import \
    SimulationTask, SimulationConfig, \
    FTV, FTA
from oricreate.gu import \
    GuConstantLength, GuDofConstraints, GuPsiConstraints, fix


def create_cp_factory():
    # begin
    from oricreate.api import MiuraOriCPFactory
    cp_factory = MiuraOriCPFactory(L_x=30,
                                   L_y=21,
                                   n_x=2,
                                   n_y=1,
                                   d_0=3.0,
                                   d_1=-3.0)
    # end
    return cp_factory

if __name__ == '__main__':

    cp_factory_task = create_cp_factory()
    cp = cp_factory_task.formed_object

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=True)
    plt.tight_layout()
    plt.show()

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()
    dof_constraints = fix(
        [1], [1]) + fix([4], [0, 1, 2]) + fix([3, 5], [2])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
    psi_max = np.pi / 4.0
    quad_psi_constraints = [([(i, 1.0)], 0) for i in range(17, 23)]
    gu_psi_constraints = \
        GuPsiConstraints(forming_task=cp_factory_task,
                         psi_constraints=quad_psi_constraints +
                         [([(12, 1.0)],
                           lambda t: -psi_max * t),
                          ])

    sim_config = SimulationConfig(goal_function_type='none',
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints,
                                      'psi': gu_psi_constraints},
                                  acc=1e-5, MAX_ITER=100)
    sim_task = SimulationTask(previous_task=cp_factory_task,
                              config=sim_config,
                              n_steps=20)

    cp = sim_task.formed_object
    cp.u[(1, 7, 8, 9), 2] = -4.0
    cp.u[(0, 2), 2] = -8.0
    sim_task.u_1

    ftv = FTV()
    ftv.add(sim_task.sim_history.viz3d_dict['node_numbers'], order=5)
    ftv.add(sim_task.sim_history.viz3d)
    ftv.add(gu_dof_constraints.viz3d)

    fta = FTA(ftv=ftv)
    fta.init_view(a=200, e=35, d=50, f=(0, 0, 0), r=0)
    fta.add_cam_move(a=200, e=34, n=5, d=50, r=0,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(0, 1, 4),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.render()
    fta.configure_traits()
