r'''

Fold the Miura ori crease pattern using psi control
---------------------------------------------------

'''

import numpy as np
from oricreate.api import \
    CreasePatternState, SimulationTask, SimulationConfig, \
    FuNodeDist, GuDofConstraints, fix, \
    FTV, FTA
from oricreate.fu import \
    FuTargetPsiValue
from oricreate.gu import \
    GuConstantLength, GuPsiConstraints
from oricreate.hu import \
    HuPsiConstraints


def create_cp_factory():
    # begin
    from oricreate.api import CustomCPFactory

    cp = CreasePatternState(X=[[-0.5, 0, 0],
                               [1, 0, 0],
                               [0, 1, 0],
                               [1, 1, 0],
                               [2, 1, 0],
                               [1, 2, 0]],
                            L=[[0, 1],
                               [0, 2],
                               [1, 3],
                               [2, 3],
                               [0, 3],
                               [1, 4],
                               [3, 4],
                               [2, 5],
                               [3, 5]],
                            F=[[0, 1, 3],
                               [0, 3, 2],
                               [1, 4, 3],
                               [2, 3, 5]
                               ])
    cp_factory = CustomCPFactory(formed_object=cp)

    # end
    return cp_factory

if __name__ == '__main__':

    cpf = create_cp_factory()
    cp = cpf.formed_object

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=True)
    plt.tight_layout()
    plt.show()

    F_u_fix = cp.F_N[0]
    dof_constraints = fix([F_u_fix[0]], [0, 1, 2]) + \
        fix([F_u_fix[1]], [1, 2]) + \
        fix([F_u_fix[2]], [2])

    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
    gu_constant_length = GuConstantLength()
    fu_node_dist = \
        FuNodeDist(forming_task=cpf,
                   L=[[4, 5]])
    sim_config = SimulationConfig(goal_function_type='total potential energy',
                                  gu={'cl': gu_constant_length,
                                      'gu': gu_dof_constraints},
                                  acc=1e-5, MAX_ITER=100)

    sim_config._fu = fu_node_dist

    sim_task = SimulationTask(previous_task=cpf,
                              config=sim_config,
                              n_steps=5)

    cp = sim_task.formed_object
    cp.u[(4, 5), 2] = 0.1
    sim_task.u_1

    ftv = FTV()
    sim_task.sim_history.viz3d['cp'].set(tube_radius=0.005)
    ftv.add(sim_task.sim_history.viz3d['cp'])
    ftv.plot()
    ftv.configure_traits()

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
