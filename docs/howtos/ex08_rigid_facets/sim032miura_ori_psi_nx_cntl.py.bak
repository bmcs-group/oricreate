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
                                   n_x=10,
                                   n_y=2,
                                   d_0=1.0,
                                   d_1=-1.0)
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

    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength()
    dof_constraints = fix(cpf.N_grid[0, 1], [1]) \
        + fix(cpf.N_grid[1, 1], [0, 1, 2]) \
        + fix(cpf.N_grid[1, (0, -1)], [2])
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
    psi_max = np.pi / 4.0
    diag_psi_constraints = [([(i, 1.0)], 0) for i in cpf.L_d_grid.flatten()]
    gu_psi_constraints = \
        GuPsiConstraints(forming_task=cpf,
                         psi_constraints=diag_psi_constraints +
                         [([(cpf.L_h_grid[1, 1], 1.0)],
                           lambda t: -psi_max * t),
                          ])

    sim_config = SimulationConfig(goal_function_type='none',
                                  gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints,
                                      'psi': gu_psi_constraints},
                                  acc=1e-5, MAX_ITER=100)
    sim_task = SimulationTask(previous_task=cpf,
                              config=sim_config,
                              n_steps=20)

    cp = sim_task.formed_object
    cp.u[cpf.N_grid[::2, :].flatten(), 2] = -0.1
    cp.u[cpf.N_grid[0, ::2].flatten(), 2] = -0.2
    sim_task.u_1

    ftv = FTV()
#    ftv.add(sim_task.sim_history.viz3d_dict['node_numbers'], order=5)
    ftv.add(sim_task.sim_history.viz3d['cp'])
    ftv.add(gu_dof_constraints.viz3d['default'])

    fta = FTA(ftv=ftv)
    fta.init_view(a=200, e=35, d=50, f=(0, 0, 0), r=0)
    fta.add_cam_move(a=200, e=34, n=5, d=50, r=0,
                     duration=10,
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.configure_traits()
