'''
Created on Aug 14, 2014

@author: rch
'''

from traits.api import \
    Property, cached_property, \
    Str, List

from mapping_task import MappingTask
from oricreate.fu import FuTargetFaces
from oricreate.gu import GuDofConstraints
from oricreate.simulation_step import \
    SimulationStep, SimulationConfig


class MapToSurface(MappingTask):

    '''The nodes of the formed are  mapped on an target face, without any
    constraints. This will be done for time_step = 0.001, so theirs only
    a little deformation.

    t_init (float): Time step which is used for the final mapping.
    default = 0.001
    '''

    name = Str('map to surface')

    target_faces = List([], input=True)
    '''List of target surfaces.
    '''

    fu_target_faces = Property(depends_on='+input')
    '''Goal function handling the distance to several target faces
    '''
    @cached_property
    def _get_fu_target_faces(self):
        return FuTargetFaces(forming_task=self, tf_lst=self.target_faces)

    dof_constraints = List([], input=True)
    '''List of dof constraints in the format
    [([( node1, dim1, factor1 ), (node2, dim2, factor2)], value ), ... ]
    defining a kinematic equation on the specified degrees of freedom 
    '''

    sim_config = Property(depends_on='+input')
    '''Configuration of the simulation step.
    '''
    @cached_property
    def _get_sim_config(self):
        gu_dofs = GuDofConstraints(forming_task=self,
                                   dof_constraints=self.dof_constraints)
        return SimulationConfig(fu=self.fu_target_faces,
                                gu={'dofs': gu_dofs})

    sim_step = Property(depends_on='+input')
    '''Simulation step object controlling the transition from time 0 to time 1
    '''
    @cached_property
    def _get_sim_step(self):
        return SimulationStep(forming_task=self,
                              config=self.sim_config, acc=1e-6)

    u_1 = Property(depends_on='+input')
    '''Resulting displacement vector
    '''
    @cached_property
    def _get_u_1(self):
        self.sim_step._solve_fmin()
        return self.formed_object.u

    x_1 = Property(depends_on='+input')
    '''Resulting position vector .
    '''
    @cached_property
    def _get_x_1(self):
        self.u_1
        return self.formed_object.x
