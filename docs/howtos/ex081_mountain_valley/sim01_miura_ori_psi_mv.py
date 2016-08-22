r'''

Fold the Miura ori crease pattern using psi control
---------------------------------------------------

'''

import numpy as np
from oricreate.api import \
    SimulationTask, SimulationConfig, \
    FTV, FTA
from oricreate.fu import \
    FuTargetPsiValue
from oricreate.gu import \
    GuConstantLength, GuPsiConstraints
from oricreate.hu import \
    HuPsiConstraints


def create_cp_factory():
    # begin
    from oricreate.api import MiuraOriCPFactory
    cp_factory = MiuraOriCPFactory(L_x=30,
                                   L_y=21,
                                   n_x=2,
                                   n_y=2,
                                   d_0=3.0,
                                   d_1=-3.0)
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

    gu_constant_length = GuConstantLength()
    psi_max = np.pi / 4.0

    diag_psi_constraints = [([(i, 1.0)], 0) for i in cpf.L_d_grid.flatten()]
    gu_psi_constraints = \
        GuPsiConstraints(forming_task=cpf,
                         psi_constraints=diag_psi_constraints
                         )
    fu_psi_target_value = \
        FuTargetPsiValue(forming_task=cpf,
                         psi_value=(cpf.L_h_grid[1, 1], lambda t: -psi_max * t)
                         )
    hu_psi_constraints = \
        HuPsiConstraints(forming_task=cpf,
                         psi_constraints=[(cpf.L_h_grid[0, 1], True),
                                          (cpf.L_v_grid[1, 0], True),
                                          (cpf.L_v_grid[1, 1], True),
                                          ])
    sim_config = SimulationConfig(goal_function_type='total potential energy',
                                  gu={'cl': gu_constant_length,
                                      'psi': gu_psi_constraints},
                                  hu={'mv': hu_psi_constraints},
                                  acc=1e-5, MAX_ITER=100)

    sim_config._fu = fu_psi_target_value

    sim_task = SimulationTask(previous_task=cpf,
                              config=sim_config,
                              n_steps=5)

    cp = sim_task.formed_object
    cp.u[cpf.N_grid[(0, -1), 1], 2] = -1.0
    sim_task.u_1

    ftv = FTV()
    ftv.add(sim_task.sim_history.viz3d_dict['node_numbers'], order=5)
    ftv.add(sim_task.sim_history.viz3d)

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
