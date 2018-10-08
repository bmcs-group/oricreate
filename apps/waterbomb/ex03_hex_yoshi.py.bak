'''
Created on Jan 20, 2016

@author: rch
'''

from traits.api import \
    Float, HasTraits, Property, cached_property, Int, \
    Instance, Array, Bool

import numpy as np
from oricreate.api import MappingTask
from oricreate.api import YoshimuraCPFactory, \
    fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, SimulationConfig, SimulationTask, \
    FTV, FTA
from oricreate.crease_pattern.crease_pattern_state import CreasePatternState
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


def get_fr(var_, L, H):
    fx = a_ * (var_ / L)**2 + b_ * (var_ / L)
    eqns = [fx.subs(var_, L), fx.subs(var_, L / 2) - H]
    ab_subs = sp.solve(eqns, [a_, b_])
    fx = fx.subs(ab_subs)
    return fx


class AddBoundaryTask(MappingTask):
    '''
    '''

    def _add_boundary_facet(self, N1, N2, dir_=-1, delta=0.1, N_start_idx=0):
        cp = self.previous_task.formed_object
        x1, x2 = cp.x_0[N1, :], cp.x_0[N2, :]
        dx = x1[:, 0] - x2[:, 0]
        dy = x1[:, 1] - x2[:, 1]
        dz = np.zeros_like(dy)
        dirvec = np.c_[dx, dy, dz]

        x4 = x2[:, :]
        x4[:, 1] += dir_ * delta
        x3 = np.copy(x4)
        x3[:, :] += dirvec * 0.82

        x_add = np.vstack([x3, x4])
        N3 = N_start_idx + np.arange(len(x3))
        N4 = N_start_idx + len(x3) + np.arange(len(x4))

        L_add = np.vstack([
            np.array([N1, N3]).T,
            np.array([N2, N3]).T,
            np.array([N3, N4]).T,
            np.array([N2, N4]).T
        ])

        F_add = np.vstack([
            np.array([N1, N3, N2]).T,
            np.array([N3, N4, N2]).T
        ])

        return x_add, L_add, F_add

    def _get_formed_object(self):
        '''attach additional facets at the obundary
        '''
        cp = self.previous_task.formed_object
        x_0, L, F = cp.x_0, cp.L, cp.F
        n_N = len(x_0)
        n_N_add = 8
        x_br, L_br, F_br = self._add_boundary_facet(
            [8, 37, 15, 43], [37, 15, 43, 20], -1, 0.1, n_N)
        x_bl, L_bl, F_bl = self._add_boundary_facet(
            [8, 31, 3, 27], [31, 3, 27, 0], -1, 0.1, n_N + n_N_add)
        x_tr, L_tr, F_tr = self._add_boundary_facet(
            [14, 42, 19, 46], [42, 19, 46, 22], 1, 0.1, n_N + 2 * n_N_add)
        x_tl, L_tl, F_tl = self._add_boundary_facet(
            [14, 36, 7, 30], [36, 7, 30, 2], 1, 0.1, n_N + 3 * n_N_add)
        x_0 = np.vstack([x_0, x_br, x_bl, x_tr, x_tl])
        L = np.vstack([L, L_br, L_bl, L_tr, L_tl])
        F = np.vstack([F, F_br, F_bl, F_tr, F_tl])
        return CreasePatternState(x_0=x_0,
                                  L=L,
                                  F=F)


class HexYoshiFormingProcess(HasTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_x = Float(3.0, auto_set=False, enter_set=True, input=True)
    L_y = Float(2.2, auto_set=False, enter_set=True, input=True)
    u_max = Float(0.1, auto_set=False, enter_set=True, input=True)
    n_fold_steps = Int(30, auto_set=False, enter_set=True, input=True)
    n_load_steps = Int(30, auto_set=False, enter_set=True, input=True)

    stiffening_boundary = Bool(False)

    ctf = Property(depends_on='+input')
    '''control target surface'''
    @cached_property
    def _get_ctf(self):
        h = 0.05
        return [r_, s_, h * t_ * s_ * (1 - s_ / self.L_y) - h * t_]

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
        u_max = self.u_max
        fa = self.factory_task
        N_h = fa.N_h
        N_i = fa.N_i
        N_v = fa.N_v
        n_x, n_y = N_h.shape
        mid_n_x = n_x / 2
        i_fixed_x = N_h[mid_n_x, (0, -1)]
        fixed_nodes_x = fix(
            i_fixed_x, (0))

        n_x, n_y = N_i.shape
        mid_n_y = n_y / 2
        i_fixed_z = np.hstack([N_i[:-1:3, mid_n_y],
                               N_i[1::3, mid_n_y]])
        i_fixed_z = np.hstack([N_i[:-1:6, mid_n_y],
                               N_i[1::6, mid_n_y]])
        fixed_nodes_z = fix(
            i_fixed_z, (2))

        i_fixed_y = N_i[(1, -2), mid_n_y]
        fixed_nodes_y = fix(
            i_fixed_y, (1))

        i_link_xh_r = N_h[2::3, :]
        i_link_xh_l = N_h[0:-2:3, :]
        i_link_xi_r = N_i[3::3, :]
        i_link_xi_l = N_i[1:-2:3, :]
        link_mid = link(
            np.hstack([i_link_xh_r.flatten(), i_link_xi_r.flatten()]),
            (0), 1.0,
            np.hstack([i_link_xh_l.flatten(), i_link_xi_l.flatten()]),
            (0), -1.0,
            lambda t: -t * u_max
        )
        sym_y = link(
            N_h[(0, -1), 0], [2], 1.0,
            N_h[(0, -1), -1], [2], -1.0
        )

        dof_constraints = fixed_nodes_x + fixed_nodes_z + fixed_nodes_y + \
            link_mid + sym_y
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='gravity potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)

        st = SimulationTask(previous_task=self.init_displ_task,
                            config=sim_config, n_steps=self.n_fold_steps)

        cp = st.formed_object

        i_down_h = N_h[1::3, :]
        i_down_i = N_i[2::3, :]

        cp.u[np.hstack([i_down_h.flatten(), i_down_i.flatten()]), 2] -= 0.1
#         cp.u[(
#             36, 48, 57, 39,
#             37, 49, 58, 40,
#             38, 50, 59, 41,
#             4, 16, 28,
#             5, 17, 29,
#             6, 18, 30,
#             7, 19, 31,
#         ), 2] -= 0.2

        return st


class HexYoshiFormingProcessFTV(FTV):

    model = Instance(HexYoshiFormingProcess)


if __name__ == '__main__':
    bsf_process = HexYoshiFormingProcess(L_x=4, L_y=12, n_x=8,
                                         n_y=6, u_max=0.8,
                                         n_fold_steps=30,
                                         n_load_steps=1)

    ftv = HexYoshiFormingProcessFTV(model=bsf_process)

    fa = bsf_process.factory_task

    it = bsf_process.init_displ_task
    ft = bsf_process.fold_task

    cp = ft.formed_object
    print 'n_dofs', cp.n_dofs
    # print ft.sim_step

    if True:
        import pylab as p
        ax = p.axes()
        fa.formed_object.plot_mpl(ax)
        p.show()

    show_init_task = False
    show_fold_task = True

    if show_init_task:
        ftv.add(it.target_faces[0].viz3d['default'])
        it.formed_object.viz3d['cp'].set(tube_radius=0.002)
        ftv.add(it.formed_object.viz3d['cp'])
        #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
        it.u_1

    if show_fold_task:
        ft.sim_history.set(anim_t_start=0, anim_t_end=10)
        ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=5)
        ft.sim_history.viz3d['cp'].set(tube_radius=0.002)
        ftv.add(ft.sim_history.viz3d['cp'])
#        ftv.add(ft.sim_history.viz3d['node_numbers'])
        ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
        ftv.add(ft.config.gu['dofs'].viz3d['default'])
        ft.u_1

    ftv.plot()
    ftv.configure_traits()
