'''
Created on Jan 20, 2016

@author: rch
'''

from oricreate.api import HPCPFactory, \
    fix, link, r_, s_, t_, MapToSurface,\
    GuConstantLength, GuDofConstraints, \
    GuPsiConstraints, SimulationConfig, SimulationTask

from oricreate.forming_tasks.forming_task import FormingTask
from oricreate.fu import \
    FuPotEngTotal
from traits.api import \
    Float, HasStrictTraits, Property, cached_property, Int, \
    Instance, Array, List, Bool

import numpy as np


def geo_transform(x_arr):
    alpha = np.pi / 4.0
    x_max = np.max(x_arr, axis=0)
    x_min = np.min(x_arr, axis=0)
    T = (x_max - x_min) / 2.0
    x_arr -= T[np.newaxis, :]

    R = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                  [np.sin(alpha), np.cos(alpha), 0],
                  [0, 0, 1]], dtype=np.float_)
    x_rot = np.einsum('ij,nj->ni', R, x_arr)
    return x_rot


class HPShellFormingProcess(HasStrictTraits):
    '''
    Define the simulation task prescribing the boundary conditions, 
    target surfaces and configuration of the algorithm itself.
    '''

    L_x = Float(10, auto_set=False, enter_set=True, input=True)
    L_y = Float(10, auto_set=False, enter_set=True, input=True)
    n_stripes = Int(2, auto_set=False, enter_set=True, input=True)

    ctf = Property(depends_on='+input')
    '''control target surface'''
    @cached_property
    def _get_ctf(self):
        return [r_, s_, t_ * 0.1 * (r_ * r_ / self.L_x +
                                    -s_ * s_ / self.L_y) + 1e-5]

    factory_task = Property(Instance(FormingTask))
    '''Factory task generating the crease pattern.
    '''
    @cached_property
    def _get_factory_task(self):

        return HPCPFactory(L_x=self.L_x, L_y=self.L_y,
                           n_stripes=self.n_stripes,
                           geo_transform=geo_transform)

    init_displ_task = Property(Instance(FormingTask))
    '''Initialization to render the desired folding branch. 
    '''
    @cached_property
    def _get_init_displ_task(self):
        cp = self.factory_task.formed_object
        return MapToSurface(previous_task=self.factory_task,
                            target_faces=[(self.ctf, cp.N)])

    psi_lines = List([10, 23, 35, 40, 7, 20, 41, 44])

    psi_max = Float(-np.pi / 2.03 * 0.8)

    fixed_z = List([9, 14])
    fixed_y = List([8, 15])
    fixed_x = List([8, 15])
    link_z = List([[8], [15]])

    n_steps = Int(5)

    fold_task = Property(Instance(FormingTask))
    '''Configure the simulation task.
    '''
    @cached_property
    def _get_fold_task(self):
        self.init_displ_task.x_1

        fixed_z = fix(self.fixed_z, (2))
        fixed_y = fix(self.fixed_y, (1))
        fixed_x = fix(self.fixed_x, (0))
        link_z = link(self.link_z[0], [2], 1, self.link_z[1], [2], -1)

        dof_constraints = fixed_x + fixed_z + fixed_y + \
            link_z
        gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)

        FN = lambda psi: lambda t: psi * t

        psi_constr = [([(i, 1.0)], FN(self.psi_max))
                      for i in self.psi_lines]

        gu_constant_length = GuConstantLength()

        gu_psi_constraints = \
            GuPsiConstraints(forming_task=self.init_displ_task,
                             psi_constraints=psi_constr)

        sim_config = SimulationConfig(goal_function_type='gravity potential energy',
                                      gu={'cl': gu_constant_length,
                                          'dofs': gu_dof_constraints,
                                          'psi': gu_psi_constraints
                                          },
                                      acc=1e-5, MAX_ITER=500,
                                      debug_level=0)
        return SimulationTask(previous_task=self.init_displ_task,
                              config=sim_config, n_steps=self.n_steps)

hp_shell_kw_2 = dict(L_x=10, L_y=10,
                     psi_lines=[
                         10, 23, 35, 40, 7, 20, 41, 44],
                     n_stripes=2,
                     n_steps=10,
                     psi_max=-np.pi / 2.5,
                     fixed_z=[9, 14],
                     fixed_y=[8, 15],
                     fixed_x=[8, 15],
                     link_z=[[8], [15]]
                     )
hp_shell_kw_4 = dict(L_x=10, L_y=10,
                     psi_lines=[
                         18, 45, 44, 69,
                         93, 98, 123, 128,
                         41, 96, 9, 34,
                         99, 102, 135, 138,
                         #                    15, 12, 63, 90,
                         #                    131, 134, 72, 103,
                     ],
                     n_stripes=4,
                     n_steps=1,
                     psi_max=-np.pi / 2.03 * 0.1,
                     fixed_z=[15, 44],
                     fixed_y=[24, 35],
                     fixed_x=[24, 35],
                     link_z=[[24], [35]]
                     )

if __name__ == '__main__':

    from oricreate.api import \
        FTV

    bsf_process = HPShellFormingProcess(**hp_shell_kw_2)

    ftv = FTV()

    fa = bsf_process.factory_task
    if False:
        import pylab as p
        ax = p.axes()
        fa.formed_object.plot_mpl(ax, nodes=True, facets=False)
        p.show()

    it = bsf_process.init_displ_task
    ft = bsf_process.fold_task

    # ftv.add(it.target_faces[0].viz3d['default'])
    ft.formed_object.viz3d['cp'].set(tube_radius=0.02)
    ftv.add(ft.sim_history.viz3d['cp'])
    #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
    ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
    ftv.add(ft.config.gu['dofs'].viz3d['default'])

    ft.u_1

    ftv.plot()
    ftv.configure_traits()
