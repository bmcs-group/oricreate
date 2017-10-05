'''
Created on Jan 20, 2016

@author: rch
'''

from math import sqrt

from traits.api import \
    Float, HasStrictTraits, Property, cached_property, Int, \
    Instance, Array, Bool, List

import numpy as np
from oricreate.api import MappingTask
from oricreate.api import YoshimuraCPFactory, \
    fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, \
    GuPsiConstraints, SimulationConfig, SimulationTask, \
    FTV, FTA
from oricreate.export import \
    InfoCadMeshExporter, ScaffoldingExporter
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.fu import \
    FuPotEngTotal
from oricreate.mapping_tasks.mask_task import MaskTask
from oricreate.simulation_tasks.simulation_history import \
    SimulationHistory
import sympy as sp


a_, b_ = sp.symbols('a,b')


class WBShellFormingProcess(HasStrictTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_x = Property

    def _get_L_x(self):
        return (self.a + 2 * self.c) * self.n_cell_x
    L_y = Property

    def _get_L_y(self):
        return self.h * 2 * self.n_cell_y

    n_fold_steps = Int(30, auto_set=False, enter_set=True, input=True)
    n_load_steps = Int(30, auto_set=False, enter_set=True, input=True)

    n_cell_x = Int(1, auto_set=False, enter_set=True, input=True)
    n_cell_y = Int(1, auto_set=False, enter_set=True, input=True)
    d_r = Float(0.1, auto_set=False, enter_set=True, input=True)
    a = Float(1.0, auto_set=False, enter_set=True, input=True)
    h = Float(1.0, auto_set=False, enter_set=True, input=True)
    c = Float(1.0, auto_set=False, enter_set=True, input=True)

    n_x = Property
    '''Number of cross cells in x direction
    '''

    def _get_n_x(self):
        return self.n_cell_x * 3

    n_y = Property
    '''Number of cross cells in x direction
    '''

    def _get_n_y(self):
        return self.n_cell_y * 2

    stiffening_boundary = Bool(False)

    ctf = Property(depends_on='+input')
    '''control target surface'''
    @cached_property
    def _get_ctf(self):
        d_r = self.d_r
        return [r_, s_,
                d_r * t_ * s_ * (1 - s_ / float(self.L_y)) - d_r * t_]

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):
        yf = YoshimuraCPFactory(L_x=self.L_x,
                                L_y=self.L_y,
                                n_x=self.n_x,
                                n_y=self.n_y)
        cp = yf.formed_object
        N_h = yf.N_h
        N_i = yf.N_i

        e = (self.a + 2 * self.c) / 3.0
        print 'e', e
        d_x = e - self.c
        print 'delta', d_x
        cp.X[N_h[1::3, :].flatten(), 0] -= d_x
        cp.X[N_h[2::3, :].flatten(), 0] += d_x
        cp.X[N_i[0::3, :].flatten(), 0] += d_x
        cp.X[N_i[2::3, :].flatten(), 0] -= d_x
        return yf

    init_displ_task = Property(Instance(FormingTask))
    '''Initialization to render the desired folding branch. 
    '''
    @cached_property
    def _get_init_displ_task(self):
        cp = self.factory_task.formed_object
        return MapToSurface(previous_task=self.factory_task,
                            target_faces=[(self.ctf, cp.N)])

    psi_max = Float(-np.pi * 0.45)

    @cached_property
    def _get_fold_angle_cntl(self):

        self.init_displ_task.x_1

        corner2_i_x = np.array([0, 0, -1, -1], dtype=np.int_)
        corner2_i_y = np.array([0, -1, 0, -1], dtype=np.int_)
        corner2_h_x = np.array([0, 0, -1, -1], dtype=np.int_)
        corner2_h_y = np.array([0, -1, 0, -1], dtype=np.int_)

        tb2_i_x = np.array([1, 1, 1], dtype=np.int_)
        tb2_i_y = np.array([0, -1, -1], dtype=np.int_)
        tb2_h_x = np.array([1, 1, 2], dtype=np.int_)
        tb2_h_y = np.array([0, -1, -1], dtype=np.int_)

        up2_i_x = np.array([0, 0, -1, -1], dtype=np.int_)
        up2_i_y = np.array([0, 1, 0, 1], dtype=np.int_)
        up2_h_x = np.array([0, 0, -1, -1], dtype=np.int_)
        up2_h_y = np.array([1, 1, 1, 1], dtype=np.int_)

        right2_i_x = np.array([2, 2, 3], dtype=np.int_)
        right2_i_y = np.array([0, 0, 0], dtype=np.int_)
        right2_h_x = np.array([3, 3, 3], dtype=np.int_)
        right2_h_y = np.array([0, 1, 0], dtype=np.int_)

        base_i_x = corner2_i_x
        base_i_y = corner2_i_y
        base_h_x = corner2_h_x
        base_h_y = corner2_h_y

        for c_x in range(0, self.n_cell_x):
            base_i_x = np.hstack([base_i_x, 3 * c_x + tb2_i_x])
            base_i_y = np.hstack([base_i_y, tb2_i_y])
            base_h_x = np.hstack([base_h_x, 3 * c_x + tb2_h_x])
            base_h_y = np.hstack([base_h_y, tb2_h_y])

        for c_x in range(0, self.n_cell_x - 1):
            base_i_x = np.hstack([base_i_x, 3 * c_x + right2_i_x])
            base_i_y = np.hstack([base_i_y, right2_i_y])
            base_h_x = np.hstack([base_h_x, 3 * c_x + right2_h_x])
            base_h_y = np.hstack([base_h_y, right2_h_y])

        for c_y in range(0, self.n_cell_y - 1):
            print 'c_y', c_y
            base_i_x = np.hstack([base_i_x, up2_i_x])
            base_i_y = np.hstack([base_i_y, c_y + up2_i_y])
            base_h_x = np.hstack([base_h_x, up2_h_x])
            base_h_y = np.hstack([base_h_y, c_y + up2_h_y])

        f = self.factory_task
        cp = f.formed_object
        m_nodes = f.N_i[base_i_x, base_i_y]
        n_nodes = f.N_h[base_h_x, base_h_y]

        psi_lines = cp.NN_L[[m_nodes], n_nodes].flatten()

        print 'psi_lines', psi_lines

        N_h = f.N_h
        N_i = f.N_i
        N_v = f.N_v
        fixed_nodes_x = fix(
            N_h[0, 0], (0))
        fixed_nodes_y = fix(
            N_h[(0, -1), 0], (1))
        fixed_nodes_z = fix(
            [N_h[0, 0], N_h[-1, 0], N_h[0, -1]], (2))

        u_max = 1.999 * self.c
        link_mid = link(
            N_i[0, 0], (0), 1.0,
            N_i[2, 0], (0), -1.0,
            lambda t: t * u_max
        )

        print '--------------------------'
        print N_i[0, 0].flatten()
        print N_i[2, 0].flatten()
        print '--------------------------'

        dof_constraints = fixed_nodes_x + fixed_nodes_z + fixed_nodes_y + \
            link_mid

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

        def FN(psi): return lambda t: psi * t
        psi_constr = [([(psi_lines[0], 1.0)], FN(self.psi_max))]
        lpsi_constr = [([(psi_lines[0], 1.0), (i, -1.0)], 0.0)
                       for i in psi_lines[1:]]

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.init_displ_task,
                             psi_constraints=lpsi_constr)

        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='none',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints,
                                          'psi': gu_psi_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)

        st = SimulationTask(previous_task=self.init_displ_task,
                            config=sim_config, n_steps=self.n_fold_steps)

        cp = st.formed_object

        N_down = np.hstack([N_h[::3, :].flatten(),
                            N_i[1::3, :].flatten()
                            ])
        print 'N_down', N_down
        N_up = np.hstack([N_i[::3, :].flatten(),
                          N_i[2::3, :].flatten(),
                          N_v[:, :].flatten()])
        print 'N_up', N_up
        cp.u[N_down, 2] -= self.d_down
        cp.u[N_up, 2] += self.d_up

        return st

    fold_angle_cntl = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def x_get_fold_angle_cntl(self):

        self.init_displ_task.x_1

        corner_i_x = np.array([0, -1], dtype=np.int_)
        corner_i_y = np.array([0, 0], dtype=np.int_)
        corner_h_x = np.array([1, -2], dtype=np.int_)
        corner_h_y = np.array([0, 0], dtype=np.int_)

        ref_i_x = 0
        ref_i_y = 0
        ref_h_x = 0
        ref_h_y = 0

        link_i_x = np.array([0, 2, 2], dtype=np.int_)
        link_i_y = np.array([-1, -1, 0], dtype=np.int_)
        link_h_x = np.array([0, 3, 3], dtype=np.int_)
        link_h_y = np.array([-1, -1, 0], dtype=np.int_)

        f = self.factory_task
        cp = f.formed_object
        m_nodes = f.N_i[corner_i_x, corner_i_y]
        n_nodes = f.N_h[corner_h_x, corner_h_y]

        psi_lines = cp.NN_L[m_nodes, n_nodes].flatten()

        print 'psi_lines', psi_lines

        rm_nodes = f.N_i[ref_i_x, ref_i_y]
        rn_nodes = f.N_h[ref_h_x, ref_h_y]
        rpsi_line = cp.NN_L[rm_nodes, rn_nodes].flatten()

        print 'rpsi_lines', rpsi_line

        lm_nodes = f.N_i[link_i_x, link_i_y]
        ln_nodes = f.N_h[link_h_x, link_h_y]
        lpsi_lines = cp.NN_L[lm_nodes, ln_nodes].flatten()

        print 'lpsi_lines', lpsi_lines

        N_h = f.N_h
        N_i = f.N_i
        N_v = f.N_v
        fixed_nodes_x = fix(
            N_h[0, 0], (0))
        fixed_nodes_y = fix(
            N_h[(0, -1), 0], (1))
        fixed_nodes_z = fix(
            [N_h[0, 0], N_h[-1, 0], N_h[0, -1]], (2))

        link_nodes = link(N_h[1, (0, -1)], 2, 1.0, N_h[2, (0, -1)], 2, -1.0)

        dof_constraints = fixed_nodes_x + fixed_nodes_z + fixed_nodes_y + \
            link_nodes

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

        def FN(psi): return lambda t: psi * t
        psi_constr = [([(i, 1.0)], FN(self.psi_max))
                      for i in psi_lines]
        lpsi_constr = [([(rpsi_line, 1.0), (i, -1.0)], 0.0)
                       for i in lpsi_lines]
        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.init_displ_task,
                             psi_constraints=psi_constr + lpsi_constr)

        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='none',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints,
                                          'psi': gu_psi_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)

        st = SimulationTask(previous_task=self.init_displ_task,
                            config=sim_config, n_steps=self.n_fold_steps)

        cp = st.formed_object

        N_down = np.hstack([N_h[::3, :].flatten(),
                            N_i[1::3, :].flatten()
                            ])
        print 'N_down', N_down
        N_up = np.hstack([N_i[::3, :].flatten(),
                          N_i[2::3, :].flatten(),
                          N_v[:, :].flatten()])
        print 'N_up', N_up
        cp.u[N_down, 2] -= self.d_down
        cp.u[N_up, 2] += self.d_up

        return st

    d_up = Float(0.01)
    d_down = Float(0.01)


class WBShellFormingProcessFTV(FTV):

    model = Instance(WBShellFormingProcess)


if __name__ == '__main__':
    kw1 = dict(a=0.36,
               c=0.13,
               h=0.55,
               d_r=0.001, d_up=0.05, d_down=0.05,
               psi_max=-np.pi * 0.52,
               n_cell_x=1, n_cell_y=2,
               n_fold_steps=20,
               n_load_steps=1)
    kw2 = dict(a=0.6,
               h=0.8,
               c=0.6,
               d_r=0.001, d_up=0.05, d_down=0.05,
               psi_max=-np.pi * 0.8,
               n_cell_x=2, n_cell_y=3,
               n_fold_steps=20,
               n_load_steps=1)
    kw3 = dict(a=0.6,
               h=0.8,
               c=0.3,
               d_r=0.0001, d_up=0.005, d_down=0.005,
               psi_max=-np.pi * 0.8,
               n_cell_x=1, n_cell_y=1,
               n_fold_steps=20,
               n_load_steps=1)

    bsf_process = WBShellFormingProcess(**kw1)

    ftv = WBShellFormingProcessFTV(model=bsf_process)

    fa = bsf_process.factory_task

    if True:
        import pylab as p
        ax = p.axes()
        fa.formed_object.plot_mpl(ax)
        p.show()

    it = bsf_process.init_displ_task

    animate = False
    show_init_task = False
    show_fold_angle_cntl = True

    fta = FTA(ftv=ftv)
    fta.init_view(a=33.4389721223,
                  e=61.453898329,
                  d=5.0,
                  f=(1.58015494765,
                     1.12671403563,
                     -0.111520325399),
                  r=-105.783218753)

    if show_init_task:
        ftv.add(it.target_faces[0].viz3d['default'])
        it.formed_object.viz3d['cp'].set(tube_radius=0.002)
        ftv.add(it.formed_object.viz3d['cp'])
        #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
        it.u_1

    if show_fold_angle_cntl:
        ft = bsf_process.fold_angle_cntl
        print 'NDOFS', ft.formed_object.n_dofs
        print ft.sim_step
        ft.sim_history.set(anim_t_start=0, anim_t_end=10)
        ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=5)
        ft.sim_history.viz3d['cp'].set(tube_radius=0.002)
        ftv.add(ft.sim_history.viz3d['cp'])
#        ftv.add(ft.sim_history.viz3d['node_numbers'])
        ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
        ftv.add(ft.config.gu['dofs'].viz3d['default'])
        ft.u_1
        fta.add_cam_move(duration=10, n=20)

    fta.plot()
    fta.configure_traits()
