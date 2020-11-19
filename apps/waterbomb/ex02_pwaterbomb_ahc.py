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
from oricreate.util.einsum_utils import \
    DELTA, EPS
import pylab as p
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
        print('e', e)
        d_x = e - self.c
        print('delta', d_x)
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
            print('c_y', c_y)
            base_i_x = np.hstack([base_i_x, up2_i_x])
            base_i_y = np.hstack([base_i_y, c_y + up2_i_y])
            base_h_x = np.hstack([base_h_x, up2_h_x])
            base_h_y = np.hstack([base_h_y, c_y + up2_h_y])

        f = self.factory_task
        cp = f.formed_object
        m_nodes = f.N_i[base_i_x, base_i_y]
        n_nodes = f.N_h[base_h_x, base_h_y]

        psi_lines = cp.NN_L[[m_nodes], n_nodes].flatten()
        print('psi_lines', psi_lines)

        cm_node = f.N_i[0, 0]
        cn_node = f.N_h[1, 1]
        cpsi_line = cp.NN_L[cm_node, cn_node]
        print('cpsi_lines', cpsi_line)

        N_h = f.N_h
        N_i = f.N_i
        N_v = f.N_v
        y_mid = N_i.shape[1] / 2
        fixed_nodes_x = fix(
            N_h[0, 0], (0))
        fixed_nodes_y = fix(
            N_h[(0, -1), 0], (1))
        fixed_nodes_z = fix(
            [N_h[0, 0], N_h[-1, 0], N_h[0, -1]], (2))

        u_max = (1.999 * self.c * self.t_max)
        link_mid = link(
            N_i[0, 0], (0), 1.0,
            N_i[2, 0], (0), -1.0,
            lambda t: t * u_max
        )

        print('--------------------------')
        print(N_i[0, 0].flatten())
        print(N_i[2, 0].flatten())
        print('--------------------------')

        dof_constraints = fixed_nodes_x + fixed_nodes_z + fixed_nodes_y  # + \
        # link_mid

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

        def FN(psi): return lambda t: psi * t
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
        print('N_down', N_down)
        N_up = np.hstack([N_i[::3, :].flatten(),
                          N_i[2::3, :].flatten(),
                          N_v[:, :].flatten()])
        print('N_up', N_up)
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
        return phi_argmax, np.array(phi_arr, dtype=np.float_)

    curvature_t = Property(Array)
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_curvature_t(self):
        u_1 = self.fold_angle_cntl.u_1
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

        print(type(phi))
        return n_uu, n_vv, phi


class WBShellFormingProcessFTV(FTV):

    model = Instance(WBShellFormingProcess)


if __name__ == '__main__':

    if True:
        L_x = 0.6
        a_arr = np.array([0.36, 0.28, 0.20, 0.12, 0.04], dtype=np.float_)
        a_arr = np.linspace(0.04, 0.36, 3)
        eta = a_arr / L_x
        c_arr = L_x * (1 - eta) / 2.0
        h = 0.55
    else:
        alpha = np.array([0.5], dtype=np.float_)
        beta = np.array([1.0], dtype=np.float_)
        h = 1
        a_arr = alpha * h
        c_arr = beta * h
    kw_arr = [dict(a=a,
                   c=c,
                   h=h,
                   d_r=0.01, d_up=0.005, d_down=0.005,
                   t_max=0.25,
                   n_cell_x=1, n_cell_y=2,
                   n_fold_steps=20,
                   n_load_steps=1) for a, c in zip(a_arr, c_arr)]

    bsf_processes = [WBShellFormingProcess(**kw)
                     for kw in kw_arr]

    fig, (ax1, ax3, ax4, ax_t_45) = p.subplots(4, 1)
    ax2 = ax1.twinx()

    t_45 = []
    c_45 = []
    z_45 = []
    for bsf_process in bsf_processes:
        ftv = WBShellFormingProcessFTV(model=bsf_process)

        fa = bsf_process.factory_task
        it = bsf_process.init_displ_task
        ft = bsf_process.fold_angle_cntl

        arg_phi, phi = bsf_process.max_slope
        n_u, n_v, c = bsf_process.curvature_t
        print(ft.t_arr)
        print(c)
        ax1.plot(ft.t_arr, c, 'b-',
                 label='curvature')
        ax2.plot(ft.t_arr, n_u, 'g-', label='height u')
        ax2.plot(ft.t_arr, n_v, 'y-', label='height v')

        arg_t_45 = np.argwhere(phi > 45.0)[0][0]
        t_45.append(ft.t_arr[arg_t_45])
        c_45.append(c[arg_t_45])
        z_45.append(n_v[arg_t_45])
        ax3.plot(ft.t_arr, phi, 'r-', label='slope')

    ax4.plot(a_arr, c_45, 'b-', label='c_45')
    ax5 = ax4.twinx()
    ax5.plot(a_arr, z_45, 'g-', label='z_45')
    ax_t_45.plot(a_arr, t_45, 'r-', label='t_45')
    p.legend()
    p.show()
    ft.sim_history.viz3d['cp'].set(tube_radius=0.002)
    ftv.add(ft.sim_history.viz3d['cp'])
    ftv.plot()
    ftv.configure_traits()

    fta = FTA(ftv=ftv)
    fta.init_view(a=33.4389721223,
                  e=61.453898329,
                  d=5.0,
                  f=(1.58015494765,
                     1.12671403563,
                     -0.111520325399),
                  r=-105.783218753)

    fta.plot()
    fta.configure_traits()
