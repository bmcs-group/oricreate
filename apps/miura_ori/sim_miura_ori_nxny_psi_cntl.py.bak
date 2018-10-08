r'''
Fold the Miura ori crease pattern using psi control
---------------------------------------------------
'''

import numpy as np
from oricreate.api import MiuraOriCPFactory
from oricreate.api import \
    SimulationTask, SimulationConfig, \
    FTV, FTA
from oricreate.forming_tasks.forming_process import FormingProcess
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.fu import \
    FuPotEngTotal
from oricreate.gu import \
    GuConstantLength, GuDofConstraints, GuPsiConstraints, fix
import traits.api as t


class MiuraOriFormingProcess(FormingProcess):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_x = t.Float(30, auto_set=False, enter_set=True, input=True)
    L_y = t.Float(21, auto_set=False, enter_set=True, input=True)
    n_x = t.Int(4, auto_set=False, enter_set=True, input=True)
    n_y = t.Int(4, auto_set=False, enter_set=True, input=True)
    d_0 = t.Float(1, auto_set=False, enter_set=True, input=True)
    d_1 = t.Float(-1, auto_set=False, enter_set=True, input=True)
    n_fold_steps = t.Int(30, auto_set=False, enter_set=True, input=True)
    n_load_steps = t.Int(2, auto_set=False, enter_set=True, input=True)

    factory_task = t.Property(t.Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @t.cached_property
    def _get_factory_task(self):
        return MiuraOriCPFactory(L_x=self.L_x,
                                 L_y=self.L_y,
                                 n_x=self.n_x,
                                 n_y=self.n_y,
                                 d_0=self.d_0,
                                 d_1=self.d_1)

    fold_task = t.Property(t.Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @t.cached_property
    def _get_fold_task(self):
        # Link the crease factory it with the constraint client
        fat = self.factory_task
        gu_constant_length = GuConstantLength()
        nx2 = 1
        ny2 = 2
        dof_constraints = fix(fat.N_grid[nx2 - 1, ny2], [1]) \
            + fix(fat.N_grid[nx2, ny2], [0, 1, 2]) \
            + fix(fat.N_grid[nx2, (ny2 - 1, ny2 + 1)], [2])
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        psi_max = np.pi * 0.4
        diag_psi_mask = np.ones_like(fat.L_d_grid, dtype=np.bool_)
        diag_psi_mask[1:-1, 1:-1] = False
        diag_psi_constraints = [([(i, 1.0)], 0)
                                for i in fat.L_d_grid[diag_psi_mask].flatten()]
        gu_psi_constraints = \
            GuPsiConstraints(forming_task=fat,
                             psi_constraints=diag_psi_constraints +
                             [([(fat.L_h_grid[nx2, ny2], 1.0)],
                               lambda t: psi_max * t),
                              ])

        sim_config = SimulationConfig(goal_function_type='none',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints,
                                          'psi': gu_psi_constraints},
                                      acc=1e-8, MAX_ITER=200)
        sim_task = SimulationTask(previous_task=fat,
                                  config=sim_config,
                                  n_steps=20)

        cp = sim_task.formed_object
        cp.u[fat.N_grid[::2, :].flatten(), 2] = -0.1
        cp.u[fat.N_grid[0, ::2].flatten(), 2] = -0.2
        return sim_task

    load_factor = t.Float(1.0, input=True, enter_set=True, auto_set=False)

    load_task = t.Property(t.Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @t.cached_property
    def _get_load_task(self):
        self.fold_task.x_1
        fat = self.factory_task

        fixed_corner_n = fat.N_grid[(0, 0, -1, -1), (1, -2, 1, -2)].flatten()
        fixed_n = fat.N_grid[(0, 0), (1, -2)].flatten()
        slide_z = fix(fat.N_grid[-1, -2], [2], lambda t: -0.8 * self.L_x * t)
        slide_x = fix(fat.N_grid[-1, -2], [0], lambda t: 0.1 * self.L_y * t)
        loaded_n = fat.N_grid[self.n_x / 2, self.n_y / 2]
        fixed_nodes_xyz = fix(fixed_n, (0, 2)) + slide_z
        #fixed_nodes_xyz = fix(fixed_corner_n, (0, 1, 2))

        dof_constraints = fixed_nodes_xyz
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()

        diag_psi_mask = np.ones_like(fat.L_d_grid, dtype=np.bool_)
        diag_psi_mask[1:-1, 1:-1] = False
        diag_lines = fat.L_d_grid[diag_psi_mask].flatten()
        diag_psi_constraints = [([(i, 1.0)], 0)
                                for i in diag_lines]
        gu_psi_constraints = \
            GuPsiConstraints(forming_task=fat,
                             psi_constraints=diag_psi_constraints)

        sim_config = SimulationConfig(
            goal_function_type='total potential energy',
            gu={'cl': gu_constant_length,
                'dofs': gu_dof_constraints,
                'psi': gu_psi_constraints
                },
            acc=1e-5, MAX_ITER=1000,
            debug_level=0
        )

        def FN(F): return lambda t: t * F

        P = 0.0 * 30.5 * self.load_factor
        F_ext_list = [(loaded_n, 2, FN(-P))]

        fu_tot_poteng = FuPotEngTotal(kappa=np.array([1.0]),
                                      F_ext_list=F_ext_list,
                                      thickness=1,
                                      exclude_lines=diag_lines
                                      )
        sim_config._fu = fu_tot_poteng
        st = SimulationTask(previous_task=self.fold_task,
                            config=sim_config, n_steps=self.n_load_steps)
        fu_tot_poteng.forming_task = st
        cp = st.formed_object
        cp.x_0 = self.fold_task.x_1
        cp.u[:, :] = 0.0
        return st


if __name__ == '__main__':

    miura_fp = MiuraOriFormingProcess(
        L_x=30, L_y=21,
        n_x=8, n_y=8, n_load_steps=10
    )
    fat = miura_fp.factory_task
    cp = fat.formed_object

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cp.plot_mpl(ax, facets=True)
        plt.tight_layout()
        plt.show()

    fot = miura_fp.fold_task
    fot.u_1

    lot = miura_fp.load_task
    lot.u_1

    ftv = FTV()
    lot.sim_history.set(anim_t_start=0, anim_t_end=50)
    lot.sim_history.viz3d['displ'].set(tube_radius=0.002,
                                       warp_scale_factor=5.0)
    ftv.add(lot.sim_history.viz3d['displ'])
    gu_dofs_viz3d = lot.config.gu['dofs'].viz3d['default']
    gu_dofs_viz3d.scale_factor = 1
    ftv.add(gu_dofs_viz3d)
    ftv.add(lot.config.fu.viz3d['default'])
    lot.config.fu.viz3d['default'].set(anim_t_start=00, anim_t_end=50)

    fta = FTA(ftv=ftv)
    fta.init_view(a=-45, e=65, d=60, f=(0, 0, 0), r=-50)
    fta.add_cam_move(a=-80, e=80, n=20, d=35, r=-45,
                     duration=10,
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.configure_traits()
