'''
Created on Jan 20, 2016

@author: rch
'''


from traits.api import \
    Float, HasStrictTraits, Property, cached_property, Int, \
    Instance, Array, Bool

import numpy as np
from oricreate.api import YoshimuraCPFactory, CustomCPFactory, \
    fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, \
    GuPsiConstraints, SimulationConfig, SimulationTask, \
    FTV, FTA, CreasePatternState
from oricreate.export import \
    InfoCadMeshExporter
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.mapping_tasks.move_task import MoveTask
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

    psi_max = Float(np.pi)

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
        fixed_nodes_x = fix(
            N_h[0, 0], (0))
        fixed_nodes_y = fix(
            N_h[(0, -1), 0], (1))
        fixed_nodes_z = fix(
            [N_h[0, 0], N_h[-1, 0], N_h[0, -1]], (2))

        print('--------------------------')
        print(N_i[0, 0].flatten())
        print(N_i[2, 0].flatten())
        print('--------------------------')

        dof_constraints = fixed_nodes_x + fixed_nodes_z + fixed_nodes_y  # + \
        # link_mid

        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

        def FN(psi): return lambda t: psi * t * self.t_max
        cpsi_constr = [([(cpsi_line, 1.0)], FN(0.99 * self.psi_max))]

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

    lower_plate = Property(Instance(FormingTask))

    @cached_property
    def _get_lower_plate(self):
        f = self.factory_task
        N_h = f.N_h
        x_1 = self.fold_angle_cntl.x_1
        x_corners = x_1[N_h[(0, 0, -1, -1), (0, -1, 0, -1)]]
        x_corners[:, 2] -= 0.005
        ft = CustomCPFactory(
            formed_object=CreasePatternState(
                X=x_corners,
                L=[[0, 1],
                   [1, 2],
                   [2, 0],
                   [1, 2],
                    [2, 3],
                    [3, 1]],
                F=[[0, 1, 2],
                   [1, 2, 3]]))
        return ft

    interlock_plate = Property(Instance(FormingTask))

    @cached_property
    def _get_interlock_plate(self):
        f = self.factory_task
        N_h, N_i = f.N_h, f.N_i
        x_1 = self.fold_angle_cntl.x_1

        N_h_F = N_h[::3, :]
        N_i_F = N_i[1::3, :]

        F_lower = np.array(
            [N_i_F, N_h_F[:-1, :-1], N_h_F[1:, :-1]], dtype=np.float_)
        F_upper = np.array(
            [N_i_F, N_h_F[:-1, 1:], N_h_F[1:, 1:]], dtype=np.float_)
        F_I = np.vstack([F_lower.reshape(3, -1).T,
                         F_upper.reshape(3, -1).T])
        L_F_I = F_I[:, [[0, 1], [1, 2], [2, 0]]]
        L_I = np.unique(L_F_I.reshape(-1, 2), axis=1)
        ft = CustomCPFactory(
            formed_object=CreasePatternState(
                X=x_1,
                L=L_I,
                F=F_I))
        return ft

    upper_plate = Property(Instance(FormingTask))

    @cached_property
    def _get_upper_plate(self):
        f = self.factory_task
        N_h = f.N_h
        x_1 = self.fold_angle_cntl.x_1
        x_corners = x_1[N_h[(1, 1, -2, -2), (0, -1, 0, -1)]]
        x_corners[:, 2] += 0.008 + 0.4
        ft = CustomCPFactory(
            formed_object=CreasePatternState(
                X=x_corners,
                L=[[0, 1],
                   [1, 2],
                   [2, 0],
                   [1, 2],
                    [2, 3],
                    [3, 1]],
                F=[[0, 1, 2],
                   [1, 2, 3]]))
        return ft

    d_up = Float(0.01)
    d_down = Float(0.01)


class WBShellFormingProcessFTV(FTV):

    model = Instance(WBShellFormingProcess)


def run_examples():
    factor = 10

    kw4 = dict(a=0.45,  # 0.36,
               c=0.17,  # 0.17,  # 0.12
               h=0.4,  # 0.4,  # 0.55
               d_r=0.1 / factor, d_up=0.01 / factor, d_down=0.04 / factor,
               #               d_r=0.01 / factor, d_up=0.01 / factor, d_down=0.04 / factor,
               #               d_r=0.05, d_up=0.005, d_down=0.01,
               t_max=1.0,
               psi_max=np.pi * 0.86,
               n_cell_x=5, n_cell_y=5,
               # n_cell_x=3, n_cell_y=3,
               n_fold_steps=50,
               n_load_steps=1)

    bsf_process = WBShellFormingProcess(**kw4)

    ftv = WBShellFormingProcessFTV(model=bsf_process)

    it = bsf_process.init_displ_task

    import pylab as p
    ax = p.axes()
    fa = bsf_process.factory_task
    fa.formed_object.plot_mpl(ax, nodes=False, lines=False, facets=False)
    p.show()

    show_init_task = False
    show_trr = True

    fta = FTA(ftv=ftv, figsize_factor=1.5)
    fta.init_view(a=33.4389721223,
                  e=61.453898329,
                  d=6.0,
                  f=(1.58015494765,
                     1.12671403563,
                     -0.111520325399),
                  r=-105.783218753)

    if show_init_task:
        ftv.add(it.target_faces[0].viz3d['default'])
        it.formed_object.viz3d['cp'].trait_set(tube_radius=0.1)
        ftv.add(it.formed_object.viz3d['cp'])
        #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
        it.u_1

    if show_trr:

        ft = bsf_process.fold_angle_cntl

        ft.sim_history.trait_set(anim_t_start=0, anim_t_end=30)
        ft.sim_history.viz3d['cp'].trait_set(tube_radius=0.005)
        ftv.add(ft.sim_history.viz3d['cp'])
        ft.formed_object.viz3d['cp'].trait_set(
            lines=True, tube_radius=0.005)
        ft.sim_history.viz3d['cp_thick'].trait_set(
            lines=False, plane_offsets=[0.006])
        ftv.add(ft.sim_history.viz3d['cp_thick'])
        ft.u_1

        lower_plate_ft = bsf_process.interlock_plate  # lower_plate
        lower_plate_ft.formed_object.trait_set(
            anim_t_start=30.0, anim_t_end=31)
        lower_plate_ft.formed_object.viz3d['cp'].trait_set(
            lines=True, tube_radius=0.005)
        ftv.add(lower_plate_ft.formed_object.viz3d['cp'])
        lower_plate_ft.formed_object.viz3d['cp_thick'].trait_set(
            lines=False, plane_offsets=[0.006])
        ftv.add(lower_plate_ft.formed_object.viz3d['cp_thick'])

        ft_center = ft.center_1
        ft_center[0] += 0.5
        ft_center[1] -= 1.0
        ft_rotate = -np.pi / 2
        mt_turn = MoveTask(previous_task=ft,
                           u_target=[0, 0, 0],
                           rotation_axis=[0, 0, 1],
                           rotation_center=ft_center,
                           rotation_angle=ft_rotate,
                           n_steps=20)
        mt_turn.sim_history.trait_set(anim_t_start=30.01, anim_t_end=40)
        mt_turn.sim_history.viz3d['cp'].trait_set(
            lines=True, tube_radius=0.005)
        ftv.add(mt_turn.sim_history.viz3d['cp'])
        mt_turn.sim_history.viz3d['cp_thick'].trait_set(
            lines=False, plane_offsets=[0.006])
        ftv.add(mt_turn.sim_history.viz3d['cp_thick'])
        mt_turn.u_1

        ft_rotate2 = -np.pi / 3
        mt_t2 = MoveTask(previous_task=lower_plate_ft,
                         u_target=[0, 0, 0],
                         rotation_axis=[0, 0, 1],
                         rotation_center=ft_center,
                         rotation_angle=ft_rotate,
                         n_steps=20)
        mt_t2.sim_history.trait_set(anim_t_start=30.01, anim_t_end=40)
        mt_t2.sim_history.viz3d['cp'].trait_set(
            lines=True, tube_radius=0.005)
        ftv.add(mt_t2.sim_history.viz3d['cp'])
        mt_t2.sim_history.viz3d['cp_thick'].trait_set(
            lines=False, plane_offsets=[0.006])
        ftv.add(mt_t2.sim_history.viz3d['cp_thick'])
        mt_t2.u_1

        mt_rotate = MoveTask(previous_task=mt_turn,
                             u_target=[0, 0, 0],
                             rotation_axis=[1, 1, 0],
                             rotation_center=ft_center,
                             rotation_angle=ft_rotate2,
                             n_steps=20)
        mt_rotate.sim_history.trait_set(anim_t_start=40.01, anim_t_end=50)
        mt_rotate.sim_history.viz3d['cp'].trait_set(
            lines=True, tube_radius=0.005)
        ftv.add(mt_rotate.sim_history.viz3d['cp'])
        mt_rotate.sim_history.viz3d['cp_thick'].trait_set(
            lines=False, plane_offsets=[0.006])
        ftv.add(mt_rotate.sim_history.viz3d['cp_thick'])
        mt_rotate.u_1

        mt_r2 = MoveTask(previous_task=mt_t2,
                         u_target=[0, 0, 0],
                         rotation_axis=[1, 1, 0],
                         rotation_center=ft_center,
                         rotation_angle=ft_rotate2,
                         n_steps=40)
        mt_r2.sim_history.trait_set(anim_t_start=40.01, anim_t_end=50)
        mt_r2.sim_history.viz3d['cp'].trait_set(
            lines=True, tube_radius=0.005)
        ftv.add(mt_r2.sim_history.viz3d['cp'])
        mt_r2.sim_history.viz3d['cp_thick'].trait_set(
            lines=False, plane_offsets=[0.006])
        ftv.add(mt_r2.sim_history.viz3d['cp_thick'], name='cp3')
        mt_r2.u_1

        if False:
            upper_plate_ft = bsf_process.upper_plate
            upper_plate_ft.formed_object.viz3d['cp'].trait_set(
                lines=False, tube_radius=0.001)
            upper_plate_ft.formed_object.trait_set(
                anim_t_start=50.0, anim_t_end=70)
    #        ftv.add(upper_plate_ft.formed_object.viz3d['cp'])

        azimuth = 120.
        elevation = 105
        f = it.center_1
        f[2] += 1
        f[0] += 0.3
        fta.init_cam_station.trait_set(
            azimuth=azimuth,
            elevation=elevation,
            distance=7,
            roll=-115.,
            focal_point=f,
        )
        f[2] -= 1
        #f = [1.49, 1.89, 0.45]
        # Turn from crease pattern view to folding position
        fta.add_cam_move(duration=10, n=10,
                         # d=6.8,
                         f=f,  # ft.center_1,
                         azimuth_move='damped',
                         elevation_move='damped',
                         distance_move='damped',
                         vot_start=0, vot_end=0)
        # Show the folding process
        fta.add_cam_move(duration=15, n=60,
                         d=6.8,
                         f=ft.center_1,
                         azimuth_move='damped',
                         elevation_move='damped',
                         distance_move='damped',
                         vot_start=0, vot_end=1)
        fta.add_cam_move(duration=5, n=20,
                         vot_start=1, vot_end=1)
        # Wait
        fta.add_cam_move(duration=1, n=10,
                         vot_start=0, vot_end=0)
        # Insert the interlocks
        f = mt_rotate.center_1
        fta.add_cam_move(duration=9, n=20,
                         f=f,
                         vot_start=0, vot_end=1)
        # Wait
        fta.add_cam_move(duration=1, n=20,
                         vot_start=0, vot_end=0)
        # Put it down again
        f = mt_rotate.center_1
        f[0] += 0.5
        fta.add_cam_move(duration=8, n=40,
                         d=4.0,
                         f=f,
                         distance_move='damped',
                         vot_start=0, vot_end=1)
        # Wait
        fta.add_cam_move(duration=1, n=20,
                         vot_start=1, vot_end=1)

    fta.plot()
    fta.render()


if __name__ == '__main__':
    run_examples()
