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

    L_x = Float(0.5, auto_set=False, enter_set=True, input=True)
    L_y = Float(0.25, auto_set=False, enter_set=True, input=True)
    n_x = Int(1, auto_set=False, enter_set=True, input=True)
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

        n_t_h = cp.N_h[:, -1].flatten()
        n_b_h = cp.N_h[:, 0].flatten()
        n_lr_h = cp.N_h[(0, -1), :].flatten()

        u_max = self.u_x
        dof_constraints = fix(n_b_h, [1], lambda t: t * u_max) + fix(n_lr_h, [2]) + \
            fix(n_t_h, [1], lambda t: t * -u_max)

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='gravity potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)

        st = SimulationTask(previous_task=self.init_displ_task,
                            config=sim_config, n_steps=self.n_steps)

        st.formed_object.u[(4, 5), 2] = 0.001

        return st

    load_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_load_task(self):
        self.fold_task.x_1

        dof_constraints = fix(
            [0, 2, 6], [2]) + fix([0, 2], [1]) + fix([6], [0])

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='total potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      use_f_du=True,
                                      acc=1e-8, MAX_ITER=1000,
                                      debug_level=0)

        FN = lambda F: lambda t: t * F

        F_ext_list = [(n, 2, FN(-1)) for n in [1, 3]]

        print('F_ext_list', F_ext_list)
        fu_tot_poteng = FuPotEngTotal(kappa=10, fu_factor=1,
                                      F_ext_list=F_ext_list)
        sim_config._fu = fu_tot_poteng
        st = SimulationTask(previous_task=self.fold_task,
                            config=sim_config, n_steps=1)
        cp = st.formed_object
        cp.x_0 = self.fold_task.x_1
        cp.u[:, :] = 0.0
        #cp.u[(4, 5), 2] = -0.001
        #cp.u[(1, 3), 2] = -0.001
        fu_tot_poteng.forming_task = st
        return st


class BikeShellterFormingProcessFTV(FTV):

    model = Instance(BarrellVaultGravityFormingProcess)


if __name__ == '__main__':
    bsf_process = BarrellVaultGravityFormingProcess(
        L_x=2.0, L_y=2.0, n_x=1, n_steps=1, u_x=0.0000001)
    #it = bsf_process.init_displ_task
    #ft = bsf_process.fold_task
    lt = bsf_process.load_task

    ftv = BikeShellterFormingProcessFTV(model=bsf_process)
#     ftv.add(it.target_faces[0].viz3d)
#     it.formed_object.viz3d.set(tube_radius=0.002)
#     ftv.add(it.formed_object.viz3d)
#     ftv.add(it.formed_object.viz3d_dict['node_numbers'], order=5)

    # lt.formed_object.viz3d.set(tube_radius=0.002)
#     ftv.add(lt.formed_object.viz3d_dict['node_numbers'], order=5)
#     ftv.add(lt.formed_object.viz3d_dict['displ'])
#     lt.config.gu['dofs'].viz3d.scale_factor = 0.5
#     ftv.add(lt.config.gu['dofs'].viz3d)

    ftv.add(lt.sim_history.viz3d['displ'])
    ftv.add(lt.config.fu.viz3d['default'])
    ftv.add(lt.config.fu.viz3d['node_load'])

    # it.u_1
    # ft.u_1
    # print 'ft_x1', ft.x_1
    lt.u_1
    cp = lt.formed_object
    print('lt_x0', cp.x_0)
    print('lt_u', cp.u)
    cp.u[:, 1] = -0.001
    print('lt.u_1', lt.u_1)

    cp = lt.formed_object
    iL_phi = cp.iL_psi - cp.iL_psi_0
    print('iL_phi',  iL_phi)

    print('lengths', cp.L_lengths)

    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()
