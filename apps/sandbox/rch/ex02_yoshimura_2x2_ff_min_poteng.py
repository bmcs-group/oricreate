from traits.api import \
    HasTraits, Float, Property, cached_property, Instance, \
    Int
import numpy as np
from oricreate.api import \
    YoshimuraCPFactory,     fix, link, r_, s_, MapToSurface,\
    GuConstantLength, GuDofConstraints, SimulationConfig, SimulationTask, \
    FTV
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.fu import \
    FuPotEngTotal


class BarrellVaultGravityFormingProcess(HasTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_x = Float(2.0, auto_set=False, enter_set=True, input=True)
    L_y = Float(1.0, auto_set=False, enter_set=True, input=True)
    n_x = Int(2, auto_set=False, enter_set=True, input=True)
    n_y = Int(2, auto_set=False, enter_set=True, input=True)
    u_x = Float(0.1, auto_set=False, enter_set=True, input=True)
    n_steps = Int(10, auto_set=False, enter_set=True, input=True)

    ctf = Property(depends_on='+input')
    '''control target surface'''
    @cached_property
    def _get_ctf(self):
        return [r_, s_, -0.1 * (r_ * (1 - r_ / self.L_x))]

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):
        return YoshimuraCPFactory(L_x=self.L_x, L_y=self.L_y,
                                  n_x=self.n_x, n_y=self.n_y)
    init_displ_task = Property(Instance(FormingTask))
    '''Initialization to render the desired folding branch. 
    '''
    @cached_property
    def _get_init_displ_task(self):
        cp = self.factory_task.formed_object
        return MapToSurface(previous_task=self.factory_task,
                            target_faces=[(self.ctf, cp.N)])

    fold_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_task(self):
        self.init_displ_task.x_1
        cp = self.factory_task

        n_l_h = cp.N_h[0, :].flatten()
        n_r_h = cp.N_h[-1, :].flatten()
        n_lr_h = cp.N_h[(0, -1), :].flatten()
        n_fixed_y = cp.N_h[(0, -1), 1].flatten()

        u_max = self.u_x
        dof_constraints = fix(n_l_h, [0], lambda t: t * u_max) + fix(n_lr_h, [2]) + \
            fix(n_fixed_y, [1]) + fix(n_r_h, [0], lambda t: t * -u_max) + \
            link(cp.N_v[0, :].flatten(), 0, 1.0,
                 cp.N_v[1, :].flatten(), 0, 1.0)

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='gravity potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)
        return SimulationTask(previous_task=self.init_displ_task,
                              config=sim_config, n_steps=self.n_steps)

    load_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_load_task(self):
        self.fold_task.x_1
        cp = self.factory_task

        n_l_h = cp.N_h[0, (0, -1)].flatten()
        n_r_h = cp.N_h[-1, (0, -1)].flatten()

        dof_constraints = fix(n_l_h, [0, 1, 2]) + fix(n_r_h, [0, 1, 2])

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='total potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      use_f_du=True,
                                      acc=1e-4, MAX_ITER=1000,
                                      debug_level=0)
        F_ext_list = [(n, 2, 100.0) for n in cp.N_h[1, :]]
        fu_tot_poteng = FuPotEngTotal(kappa=np.array([10]),
                                      F_ext_list=F_ext_list)
        sim_config._fu = fu_tot_poteng
        st = SimulationTask(previous_task=self.fold_task,
                            config=sim_config, n_steps=1)
        cp = st.formed_object
        cp.x_0 = self.fold_task.x_1
        cp.u[:, :] = 0.0
        fu_tot_poteng.forming_task = st
        return st


class BikeShellterFormingProcessFTV(FTV):

    model = Instance(BarrellVaultGravityFormingProcess)


if __name__ == '__main__':
    bsf_process = BarrellVaultGravityFormingProcess(
        L_x=2.0, n_x=2, n_steps=1, u_x=0.1)
    it = bsf_process.init_displ_task
    ft = bsf_process.fold_task
    lt = bsf_process.load_task

    it.u_1
    ft.u_1

    ftv = BikeShellterFormingProcessFTV(model=bsf_process)
#     ftv.add(it.target_faces[0].viz3d)
#     it.formed_object.viz3d.set(tube_radius=0.002)
#     ftv.add(it.formed_object.viz3d)
#     ftv.add(it.formed_object.viz3d_dict['node_numbers'], order=5)
    lt.formed_object.viz3d.set(tube_radius=0.002)
    ftv.add(lt.formed_object.viz3d_dict['node_numbers'], order=5)
    ftv.add(lt.formed_object.viz3d)
    ftv.add(lt.formed_object.viz3d_dict['displ'])
    lt.config.gu['dofs'].viz3d.scale_factor = 0.5
    ftv.add(lt.config.gu['dofs'].viz3d)

    ftv.add(lt.config.fu.viz3d)
    ftv.add(lt.config.fu.viz3d_dict['node_load'])

    print('ft_x1', ft.x_1)
    cp = lt.formed_object
    print('lt_x0', cp.x_0)
    print('lt_u', cp.u)
    cp.u[(2, 3), 2] = -0.001
    print('lt.u_1', lt.u_1)

    cp = lt.formed_object
    iL_phi = cp.iL_psi2 - cp.iL_psi_0
    print('phi',  iL_phi)

    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()

#     n_cam_move = 40
#     fta = FTA(ftv=ftv)
#     fta.init_view(a=45, e=60, d=7, f=(0, 0, 0), r=-120)
#     fta.add_cam_move(a=60, e=70, n=n_cam_move, d=6, r=-120,
#                      duration=10,
#                      vot_fn=lambda cmt: np.linspace(0.01, 0.5, n_cam_move),
#                      azimuth_move='damped',
#                      elevation_move='damped',
#                      distance_move='damped')
#     fta.add_cam_move(a=80, e=80, d=4, n=n_cam_move, r=-132,
#                      duration=10,
#                      vot_fn=lambda cmt: np.linspace(0.5, 1.0, n_cam_move),
#                      azimuth_move='damped',
#                      elevation_move='damped',
#                      distance_move='damped')
#
#     fta.plot()
#     fta.render()
#     fta.configure_traits()
