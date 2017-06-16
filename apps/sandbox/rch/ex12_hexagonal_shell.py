'''
Created on Jan 20, 2016

@author: rch
'''

from traits.api import \
    Float, HasStrictTraits, Property, cached_property, Int, \
    Instance, Array, List, Bool

import numpy as np
from oricreate.api import HexagonalCPFactory, \
    fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, \
    GuPsiConstraints, SimulationConfig, SimulationTask
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.fu import \
    FuPotEngTotal, FuPotEngGravity


def geo_transform(x_arr):
    alpha = np.pi / 4.0
    L_x = 12.0
    L_y = 4.0
    x_max = np.max(x_arr, axis=0)
    x_min = np.min(x_arr, axis=0)
    T = (x_max - x_min) / 2.0
    x_arr -= T[np.newaxis, :]

    R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                  [np.sin(alpha), np.cos(alpha), 0],
                  [0, 0, 1]], dtype=np.float_)
    x_rot = np.einsum('ij,nj->ni', R, x_arr)
    x_rot[:, 0] *= L_x / 2.0
    x_rot[:, 1] *= L_y / 2.0
    return x_rot


class HexagonalShellFormingProcess(HasStrictTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_x = Float(10.0, auto_set=False, enter_set=True, input=True)
    L_y = Float(4.0, auto_set=False, enter_set=True, input=True)
    n_stripes = Int(2, auto_set=False, enter_set=True, input=True)

    ctf = Property(depends_on='+input')
    '''control target surface'''
    @cached_property
    def _get_ctf(self):
        return [r_, s_, t_ * (r_ * r_ / self.L_x +
                              -s_ * s_ / self.L_y) + 1e-5]

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):

        return HexagonalCPFactory(n_seg=self.n_stripes,
                                  geo_transform=geo_transform)

    init_displ_task = Property(Instance(FormingTask))
    '''Initialization to render the desired folding branch. 
    '''
    @cached_property
    def _get_init_displ_task(self):
        cp = self.factory_task.formed_object
        return MapToSurface(previous_task=self.factory_task,
                            target_faces=[(self.ctf, cp.N)])

    fixed_z = List([2, 10, 6, 14])
    fixed_y = List([2, 6])
    fixed_x_plus = List([2, 10])
    fixed_x_minus = List([6, 14])

    n_steps = Int(5)

    kappa = Float(1.0)
    rho = Float(1.0)
    MAXITER = Int(500)

    u_max = Float(0.02)

    fold_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_task(self):
        L_rigid = self.factory_task.L_rigid
        N_up = self.factory_task.N_up
        N_down = self.factory_task.N_down
        N_x_sym = self.factory_task.N_x_sym[::2]
        print 'N_x_sym', N_x_sym
#        N_x_sym = []  # self.factory_task.N_x_sym[[0, -1]]

        print 'n_dofs', self.factory_task.formed_object.n_dofs

        self.init_displ_task.x_1

        fixed_z = fix(self.fixed_z, (2))
        fixed_y = fix(self.fixed_y, (1))
        fixed_x = fix(N_x_sym, (0))
        fixed_x_plus = fix(self.fixed_x_plus, (0),
                           lambda t: t * self.u_max)
        fixed_x_minus = fix(self.fixed_x_minus, (0),
                            lambda t: -t * self.u_max)

        dof_constraints = fixed_x_minus + \
            fixed_x_plus + fixed_x + fixed_z + fixed_y
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

        def FN(psi): return lambda t: psi * t

        psi_constr = [([(i, 1.0)], FN(0))
                      for i in L_rigid]

        gu_constant_length = GuConstantLength()

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.init_displ_task,
                             psi_constraints=psi_constr)

        sim_config = SimulationConfig(goal_function_type='gravity potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints,
                                          'psi': gu_psi_constraints
                                          },
                                      acc=1e-4, MAX_ITER=self.MAXITER,
                                      debug_level=0)
        st = SimulationTask(previous_task=self.init_displ_task,
                            config=sim_config, n_steps=self.n_steps,
                            record_iter=False)
        cp = st.formed_object
        cp.u[N_up, 2] += 0.01
        cp.u[N_down, 2] -= 0.01
        return st

    fold_deform_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_deform_task(self):
        ft = self.factory_task

        self.init_displ_task.x_1

        fixed_z = fix(self.fixed_z, (2))
        fixed_y = fix(self.fixed_y, (1))
        fixed_x = fix(self.fixed_x, (0))
        link_z = link(self.link_z[0], [2], 1, self.link_z[1], [2], -1)

        dof_constraints = fixed_x + fixed_z + fixed_y + \
            link_z
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

        def FN(psi): return lambda t: psi * t

        psi_constr = [([(i, 1.0)], FN(self.psi_max))
                      for i in self.psi_lines]

        gu_constant_length = GuConstantLength()

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=ft,
                             psi_constraints=psi_constr)

        sim_config = SimulationConfig(goal_function_type='gravity potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints,
                                          'psi': gu_psi_constraints
                                          },
                                      acc=1e-5, MAX_ITER=self.MAXITER,
                                      debug_level=0)

        fu_tot_poteng = FuPotEngTotal(kappa=np.array([self.kappa]),
                                      F_ext_list=[],
                                      rho=self.rho,
                                      exclude_lines=self.psi_lines)

        sim_config._fu = fu_tot_poteng

        st = SimulationTask(previous_task=self.init_displ_task,
                            config=sim_config, n_steps=self.n_steps)

#
#         st = SimulationTask(previous_task=ft,
#                             config=sim_config, n_steps=self.n_steps)
        fu_tot_poteng.forming_task = st
#         gu_psi_constraints.forming_task = st
#
#         cp = st.formed_object
#         cp.u = it.u_1

        return st


shell_kw_2 = dict(L_x=12, L_y=4,
                  kappa=0.00001,
                  n_stripes=2,
                  n_steps=4,
                  rho=10,
                  MAXITER=1000,
                  #psi_max=-np.pi / 2.03 * 0.5,
                  fixed_z=[2, 10, 6, 14],
                  fixed_y=[2, 6],
                  fixed_x_plus=[2, 10],
                  fixed_x_minus=[6, 14],
                  )

shell_kw_4 = dict(L_x=5, L_y=2.5,
                  kappa=0.00001,
                  n_stripes=4,
                  n_steps=1,
                  rho=10,
                  MAXITER=10,
                  #psi_max=-np.pi / 2.03 * 0.5,
                  fixed_z=[8, 88, 72, 152],
                  fixed_y=[8, 72],
                  fixed_x_plus=[8, 88],
                  fixed_x_minus=[72, 152],
                  )

if __name__ == '__main__':

    from oricreate.api import \
        FTV

    bsf_process = HexagonalShellFormingProcess(**shell_kw_4)

    ftv = FTV()

    fa = bsf_process.factory_task
    if False:
        import pylab as p
        ax = p.axes()
        fa.formed_object.plot_mpl(ax, nodes=True, facets=False)
        p.show()

    it = bsf_process.init_displ_task

    print 'n_dofs', it.formed_object.n_dofs
    ft = bsf_process.fold_task
    #fd = bsf_process.fold_deform_task

    show_init_task = False
    show_fold_task = True
    show_deform_fold_task = False

    if show_init_task:
        # ftv.add(it.target_faces[0].viz3d['default'])
        it.formed_object.viz3d['cp'].set(tube_radius=0.02)
        ftv.add(it.formed_object.viz3d['cp'])
        #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
        it.u_1
    if show_fold_task:
        # ftv.add(it.target_faces[0].viz3d['default'])
        ft.sim_history.viz3d['cp'].set(tube_radius=0.02)
        ftv.add(ft.sim_history.viz3d['cp'])
        #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
        ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
        ftv.add(ft.config.gu['dofs'].viz3d['default'])

        ft.u_1

    if show_deform_fold_task:
        # ftv.add(it.target_faces[0].viz3d['default'])
        fd.sim_history.viz3d['cp'].set(tube_radius=0.02)
        ftv.add(ft.sim_history.viz3d['cp'])
        #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
        fd.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
        ftv.add(fd.config.gu['dofs'].viz3d['default'])

        fd.u_1

    ftv.plot()
    ftv.configure_traits()
