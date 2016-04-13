'''
Created on Jan 20, 2016

@author: rch
'''

from traits.api import \
    Float, HasTraits, Property, cached_property, Int, \
    Instance

import numpy as np
from oricreate.api import YoshimuraCPFactory, \
    fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, SimulationConfig, \
    SimulationStep, SimulationTask, \
    FTV, FTA, FuTF
from oricreate.crease_pattern import \
    CreasePatternNormalsViz3D, CreasePatternBasesViz3D
from oricreate.crease_pattern.crease_pattern_viz3d import \
    CreasePatternThickViz3D
from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.mapping_tasks.mask_task import MaskTask
from oricreate.viz3d.viz3d import Viz3D
import sympy as sp


a_, b_ = sp.symbols('a,b')


def get_fr(var_, L, H):
    fx = a_ * (var_ / L)**2 + b_ * (var_ / L)
    eqns = [fx.subs(var_, L), fx.subs(var_, L / 2) - H]
    ab_subs = sp.solve(eqns, [a_, b_])
    fx = fx.subs(ab_subs)
    return fx


class BikeShellterFormingProcess(HasTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_x = Float(3.0, auto_set=False, enter_set=True, input=True)
    L_y = Float(2.2, auto_set=False, enter_set=True, input=True)
    n_x = Int(4, auto_set=False, enter_set=True, input=True)
    n_y = Int(5, auto_set=False, enter_set=True, input=True)
    u_x = Float(0.1, auto_set=False, enter_set=True, input=True)
    n_steps = Int(10, auto_set=False, enter_set=True, input=True)

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):
        return YoshimuraCPFactory(L_x=self.L_x, L_y=self.L_y,
                                  n_x=self.n_x, n_y=self.n_y)

    init_task = Property(Instance(FormingTask))
    '''Initialization to render the desired folding branch. 
    '''
    @cached_property
    def _get_init_task(self):
        cpf = self.factory_task
        n_arr = np.hstack([cpf.N_h[:, :].flatten(),
                           cpf.N_i[:, :].flatten()
                           ])
        return MapToSurface(previous_task=self.factory_task,
                            target_faces=[(self.ctf, n_arr)])

    ctf = Property(depends_on='+input')
    '''control target surface'''
    @cached_property
    def _get_ctf(self):
        return [r_, s_, t_ * r_ * (1 - r_ / self.L_x) + 0.000015]

    fold_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_task(self):
        cpf = self.factory_task
        fixed_node = fix(cpf.N_h[0, -1], (0, 1, 2))
        planar_front_boundary = link(cpf.N_h[0, 0], 1, 1.0,
                                     cpf.N_h[1:, 0], 1, -1.0)
        planar_back_boundary = link(cpf.N_h[0, -1], 1, 1.0,
                                    cpf.N_h[1:, -1], 1, -1.0)
        linked_left_boundary_x = link(cpf.N_h[0, 0], 0, 1.0,
                                      cpf.N_h[0, 1:], 0, -1.0)
        linked_left_boundary_z = link(cpf.N_h[0, 0], 2, 1.0,
                                      cpf.N_h[0, 1:], 2, -1.0)
        linked_left_and_right_z = link(cpf.N_v[0, :], 2, 1.0,
                                       cpf.N_v[1, :], 2, -1.0)
        linked_right_boundary_x = link(cpf.N_v[-1, 0], 0, 1.0,
                                       cpf.N_v[-1, 1:], 0, -1.0)
        cntrl_displ = [
            ([(cpf.N_h[-1, 1], 0, 1.0)], lambda t: t * self.u_x)]
        dof_constraints = fixed_node + \
            planar_front_boundary + \
            planar_back_boundary + \
            linked_left_boundary_x + \
            linked_left_boundary_z + \
            linked_left_and_right_z + \
            linked_right_boundary_x + \
            cntrl_displ
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
        gu_constant_length = GuConstantLength()
        sim_config = SimulationConfig(goal_function_type='none',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints},
                                      acc=1e-5, MAX_ITER=500)
        return SimulationTask(previous_task=self.init_task,
                              config=sim_config, n_steps=self.n_steps)


class BikeShellterFormingProcessFTV(FTV):

    model = Instance(BikeShellterFormingProcess)


if __name__ == '__main__':
    bsf_process = BikeShellterFormingProcess(L_x=3.0, L_y=2.41, n_x=4,
                                             n_y=12, u_x=-2.0, n_steps=10)

    ftv = BikeShellterFormingProcessFTV(model=bsf_process)

    it = bsf_process.init_task
    ft = bsf_process.fold_task
    ftv.add(it.target_faces[0].viz3d)

    it.u_1
    ft.u_1

    ftv.add(ft.sim_history.viz3d_dict['node_numbers'], order=5)
    ftv.add(ft.sim_history.viz3d)
    ftv.add(ft.config.gu['dofs'].viz3d)

    # @todo: change the surface to a reasonable value.

    ftv.plot()
    ftv.update(vot=1, force=True)
    ftv.show()

    fta = FTA(ftv=ftv)
    fta.init_view(a=45, e=50, d=8, f=(0, 0, 0), r=0)
    fta.add_cam_move(a=45, e=50, n=50, d=8, r=0,
                     duration=1,
                     vot_fn=lambda cmt: np.linspace(0, 1, 10),
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.render()
    fta.configure_traits()
