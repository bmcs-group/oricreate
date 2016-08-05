from traits.api import \
    HasTraits, Float, Property, cached_property, Instance, \
    Int
import numpy as np
from oricreate.api import \
    YoshimuraCPFactory,     fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, SimulationConfig, SimulationTask, \
    FTV, FTA
from oricreate.crease_pattern.crease_pattern_state import CreasePatternState
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.fu import \
    FuPotEngTotal
from oricreate.mapping_tasks.mask_task import MaskTask


class BarrellVaultGravityFormingProcess(HasTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_x = Float(3.0, auto_set=False, enter_set=True, input=True)
    L_y = Float(2.0, auto_set=False, enter_set=True, input=True)
    n_x = Int(3, auto_set=False, enter_set=True, input=True)
    n_y = Int(4, auto_set=False, enter_set=True, input=True)
    u_x = Float(0.5, auto_set=False, enter_set=True, input=True)
    n_steps = Int(30, auto_set=False, enter_set=True, input=True)

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
        x_1 = self.init_displ_task.x_1
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
        n_fixed_y = cp.N_h[(0, -1), 1].flatten()
        n_scheitel = cp.N_h[2, :]
        n_aussen = cp.N_v

        dof_constraints = fix(
            n_l_h, [0, 2]) + fix(n_r_h, [0, 2]) + fix(n_fixed_y, [0, 1])

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='total potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-10, MAX_ITER=1000,
                                      debug_level=0)
        FN = lambda F: lambda t: t * F
        F_max = 2
        F_min = 1

        F_ext_list = [(7, 2, FN(F_max)), (6, 2, FN(F_min)), (8, 2, FN(F_min))]
        fu_tot_poteng = FuPotEngTotal(kappa=np.array([4.97]),
                                      F_ext_list=F_ext_list)
        sim_config._fu = fu_tot_poteng
        st = SimulationTask(previous_task=self.fold_task,
                            config=sim_config, n_steps=10)
        cp = st.formed_object
        cp.L = np.vstack([cp.L, np.array([[6, 7], [7, 8]], dtype='int_')])

        cp.x_0 = self.fold_task.x_1
        cp.u[:, :] = 0.0
        cp.u[[6, 7, 8], 2] = 0.0001
        fu_tot_poteng.forming_task = st
        return st


class BikeShellterFormingProcessFTV(FTV):

    model = Instance(BarrellVaultGravityFormingProcess)


if __name__ == '__main__':
    bsf_process = BarrellVaultGravityFormingProcess(
        L_x=0.63, n_x=4, L_y=0.408, n_y=4, n_steps=5, u_x=0.125 / 2.0)
    it = bsf_process.init_displ_task

    ft = bsf_process.fold_task
    lt = bsf_process.load_task

    print 'ndofs', lt.formed_object.n_dofs - lt.formed_object.n_L
    ftv = BikeShellterFormingProcessFTV(model=bsf_process)

    animate = False
    show_init_task = False
    show_fold_task = False
    show_load_task = True

    if show_init_task:
        ftv.add(it.target_faces[0].viz3d)
        it.formed_object.viz3d.set(tube_radius=0.002)
        ftv.add(it.formed_object.viz3d)
        ftv.add(it.formed_object.viz3d_dict['node_numbers'], order=5)
        it.u_1

    if show_fold_task:
        ft.sim_history.viz3d.set(tube_radius=0.002)
        ftv.add(ft.sim_history.viz3d)
        ft.config.gu['dofs'].viz3d.scale_factor = 0.5
        ftv.add(ft.config.gu['dofs'].viz3d)
        ft.u_1

    if show_load_task == True:
        lt = bsf_process.load_task
        lt.formed_object.viz3d.set(tube_radius=0.2)

        ftv.add(lt.formed_object.viz3d_dict['displ'])
        lt.config.gu['dofs'].viz3d.scale_factor = 0.2
        ftv.add(lt.config.gu['dofs'].viz3d)
        ftv.add(lt.config.fu.viz3d)

        ftv.add(lt.config.fu.viz3d_dict['node_load'])
#         print 'u_', lt.u_1[6, 2]
#         print 'u_', lt.u_1[7, 2]
#         print 'u_', lt.u_1[8, 2]
        n_max_uz = np.argmax(lt.u_1[:, 2])
        n_max_uy = np.argmax(lt.u_1[:, 1])
        n_max_ux = np.argmax(lt.u_1[:, 0])

        print 'node max_uz', n_max_uz
        print 'uz_max', lt.u_1[n_max_uz, 2]
        print 'node max_uy', n_max_uy
        print 'uy_max', lt.u_1[n_max_uy, 2]
        print 'node max_ux', n_max_ux
        print 'ux_max', lt.u_1[n_max_ux, 2]

        cp = lt.formed_object
        iL_phi = cp.iL_psi2 - cp.iL_psi_0
        iL_m = lt.config._fu.kappa * iL_phi
        print 'moment', iL_m
        print 'max moment', np.max(np.fabs(iL_m))

        ftv.plot()
        ftv.update(vot=1, force=True)
        ftv.show()

    n_cam_move = 20
    fta = FTA(ftv=ftv)
    fta.init_view(a=45, e=60, d=7, f=(0, 0, 0), r=-120)
    fta.add_cam_move(a=60, e=70, n=n_cam_move, d=8, r=-120,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(0.01, 0.5, n_cam_move),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')
    fta.add_cam_move(a=80, e=80, d=7, n=n_cam_move, r=-132,
                     duration=10,
                     vot_fn=lambda cmt: np.linspace(0.5, 1.0, n_cam_move),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.render()
    fta.configure_traits()
