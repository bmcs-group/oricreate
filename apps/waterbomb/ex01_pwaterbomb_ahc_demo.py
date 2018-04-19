'''
Created on Jan 20, 2016

@author: rch
'''


from traits.api import \
    Float, HasStrictTraits, Property, cached_property, Int, \
    Instance, Array, Bool

import numpy as np
from oricreate.api import YoshimuraCPFactory, \
    fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, \
    GuPsiConstraints, SimulationConfig, SimulationTask, \
    FTV, FTA
from oricreate.export import \
    InfoCadMeshExporter
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.util.einsum_utils import \
    EPS
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

    t_max = Float(1.0)

    fold_angle_cntl = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
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

        cm_node = f.N_i[0, 0]
        cn_node = f.N_h[1, 1]
        cpsi_line = cp.NN_L[cm_node, cn_node]
        print 'cpsi_lines', cpsi_line

        N_h = f.N_h
        N_i = f.N_i
        N_v = f.N_v
        fixed_nodes_x = fix(
            N_h[0, 0], (0))
        fixed_nodes_y = fix(
            N_h[(0, -1), 0], (1))
        fixed_nodes_z = fix(
            [N_h[0, 0], N_h[-1, 0], N_h[0, -1]], (2))

        print '--------------------------'
        print N_i[0, 0].flatten()
        print N_i[2, 0].flatten()
        print '--------------------------'

        dof_constraints = fixed_nodes_x + fixed_nodes_z + fixed_nodes_y  # + \
        # link_mid

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

        def FN(psi): return lambda t: psi * t * self.t_max
        cpsi_constr = [([(cpsi_line, 1.0)], FN(0.99 * np.pi))]

        lpsi_constr = [([(psi_lines[0], 1.0), (i, -1.0)], 0.0)
                       for i in psi_lines[1:]]

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.init_displ_task,
                             psi_constraints=lpsi_constr + cpsi_constr)

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
        cp.u[:, 2] += self.d_down

        return st

    fold_kinem_cntl = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_kinem_cntl(self):

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

        psi_lines = cp.NN_L[m_nodes, n_nodes].flatten()
        print 'psi_lines', psi_lines

        cm_nodes = [f.N_i[0, 0], f.N_i[-1, 1]]
        cn_nodes = [f.N_h[1, 1], f.N_h[2, 1]]
        cpsi_lines = cp.NN_L[cm_nodes, cn_nodes]
        print 'cpsi_lines', cpsi_lines

        N_h = f.N_h
        N_i = f.N_i
        N_v = f.N_v
        fixed_nodes_x = fix(
            N_i[1, (0, 1)], (0))
        fixed_nodes_y = fix(
            [N_h[1, 1].flatten()], (1))
        fixed_nodes_z = fix(
            list(N_v[[0, 0, -1], [0, -1, 0]].flatten()), (2)
        )
        link_nodes_yz = link(
            list(N_v[[0, 0], [0, -1]].flatten()) +
            list(N_h[[0, 0], [0, -1]].flatten()), (1, 2), 1.0,
            list(N_i[[0, 0], [0, -1]].flatten()) +
            list(N_h[[-1, -1], [0, -1]].flatten()), (1, 2), -1.0,
        )
        link_nodes_z = link(
            N_h[1, 1], 2, 1.0,
            N_h[2, 1], 2, -1.0,
        )

        dof_constraints = fixed_nodes_x + fixed_nodes_z + fixed_nodes_y + \
            link_nodes_yz + link_nodes_z

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

        def FN(psi): return lambda t: psi * t
        cpsi_constr = [([(cpsi_line, 1.0)], FN(0.99 * np.pi))
                       for cpsi_line in cpsi_lines]

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.init_displ_task,
                             psi_constraints=cpsi_constr)

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
        cp.u[:, 2] += self.d_down

        return st

    d_up = Float(0.01)
    d_down = Float(0.01)

    max_slope = Property(Array)

    @cached_property
    def _get_max_slope(self):
        u_t = self.fold_angle_cntl.sim_history.u_t
        phi_argmax = []
        phi_arr = []
        cp = ft.formed_object
        for u in u_t:
            cp.u = u
            n = cp.norm_F_normals
            n1, n2, n3 = n.T
            n12 = np.sqrt(n1**2 + n2**2)
            phi = np.pi / 2.0 - np.arctan(n3 / n12)
            i = np.argmax(phi)
            phi_argmax.append(i)
            phi_arr.append(180.0 / np.pi * phi[i])
        return phi_argmax, phi_arr

    curvature_t = Property(Array)
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_curvature_t(self):
        self.fold_angle_cntl.u_1
        f = self.factory_task
        x_t = self.fold_angle_cntl.sim_history.x_t
        u = x_t[:, f.N_h[1, 0], :] - x_t[:, f.N_h[0, 0], :]
        v = x_t[:, f.N_i[0, 0], :] - x_t[:, f.N_i[1, 0], :]

        u[:, 0] = 0
        v[:, 0] = 0
        uxv = np.einsum('...i,...j,...ijk', u, v, EPS)
        s_uxv = -np.sign(uxv[:, 0])
        n_uxv = np.sqrt(np.einsum('...i,...i', uxv, uxv))
        n_uu = np.sqrt(np.einsum('...i,...i', u, u))
        n_vv = np.sqrt(np.einsum('...i,...i', v, v))
        phi = np.arcsin(s_uxv * n_uxv / (n_uu * n_vv))

        print n_uu
        return n_uu, n_vv, phi

    def generate_fe_mesh(self, t):
        ft = self.fold_angle_cntl
        cp = ft.formed_object
        u_t = ft.sim_history.u_t
        arg_t = np.argwhere(ft.t_arr > t)[0][0]
        cp.u = u_t[arg_t]

        me = InfoCadMeshExporter(forming_task=ft, n_l_e=4)
        me.write()
        X, F = me._get_geometry()
        x, y, z = X.T
        import mayavi.mlab as m
        me.plot_mlab(m)
        m.show()


class WBShellFormingProcessFTV(FTV):

    model = Instance(WBShellFormingProcess)


factor = 10
if __name__ == '__main__':
    kw1 = dict(a=0.36,
               c=0.12,
               h=0.55,
               d_r=0.01 / factor, d_up=0.01 / factor, d_down=0.02 / factor,
               # d_r=0.0005, d_up=0.0005, d_down=0.01,
               t_max=1.0,
               n_cell_x=4, n_cell_y=4,
               n_fold_steps=5 * factor,
               n_load_steps=1)
    kw3 = dict(a=2,
               h=3,
               c=2,
               d_r=0.01, d_up=0.01, d_down=0.02,
               t_max=1.0,
               n_cell_x=1, n_cell_y=1,
               n_fold_steps=40,
               n_load_steps=1)

    bsf_process = WBShellFormingProcess(**kw1)

    ftv = WBShellFormingProcessFTV(model=bsf_process)

    fa = bsf_process.factory_task

    if True:
        import pylab as p
        ax = p.axes()
        fa.formed_object.plot_mpl(ax, nodes=False, lines=False, facets=False)
        p.show()

    it = bsf_process.init_displ_task

    animate = False
    show_init_task = False
    show_fold_angle_cntl = True
    show_fold_kinem_cntl = False

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
        it.formed_object.viz3d['cp'].set(tube_radius=0.1)
        ftv.add(it.formed_object.viz3d['cp'])
        #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
        it.u_1

    if show_fold_angle_cntl:
        ft = bsf_process.fold_angle_cntl

        print ft.sim_step

        ft.sim_history.set(anim_t_start=0, anim_t_end=10)
#        ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=5)
        ft.sim_history.viz3d['cp'].set(tube_radius=0.002)
        ftv.add(ft.sim_history.viz3d['cp'])
#        ftv.add(ft.sim_history.viz3d['node_numbers'])
#        ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
#        ftv.add(ft.config.gu['dofs'].viz3d['default'])
        ft.u_1

        bsf_process.generate_fe_mesh(0.5)

        fta.add_cam_move(duration=10, n=20)

        arg_phi, phi = bsf_process.max_slope

        fig, (ax1, ax3) = p.subplots(2, 1, sharex=True)
        ax2 = ax1.twinx()
        n_u, n_v, c = bsf_process.curvature_t
        ax1.plot(ft.t_arr, c, 'b-',
                 label='curvature')
        ax2.plot(ft.t_arr, n_u, 'g-', label='height u')
        ax2.plot(ft.t_arr, n_v, 'y-', label='height v')
        ax3.plot(ft.t_arr, phi, 'r-', label='slope')

        # p.show()

    if show_fold_kinem_cntl:
        ft = bsf_process.fold_kinem_cntl

        print ft.sim_step

        ft.sim_history.set(anim_t_start=0, anim_t_end=10)
        ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=5)
        ft.sim_history.viz3d['cp'].set(tube_radius=0.002)
        ftv.add(ft.sim_history.viz3d['cp'])
        ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
        ftv.add(ft.config.gu['dofs'].viz3d['default'])
        ft.u_1

        fta.add_cam_move(duration=10, n=20)

    fta.plot()
    fta.configure_traits()
