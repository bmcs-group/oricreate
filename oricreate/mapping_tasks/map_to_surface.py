'''
Created on Aug 14, 2014

@author: rch
'''

from traits.api import \
    Property, cached_property, \
    Str, List, Instance

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

    fu_target_faces = Instance(FuTargetFaces, input=True)
    '''List of target surfaces.
    '''

    target_faces = Property
    '''Goal function handling the distance to several target faces
    '''

    def _set_target_faces(self, values):
        self.fu_target_faces = FuTargetFaces(forming_task=self)
        self.fu_target_faces.target_faces = values

    def _get_target_faces(self):
        return self.fu_target_faces

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
                                gu={'dofs': gu_dofs}, acc=1e-6)

    sim_step = Property(depends_on='+input')
    '''Simulation step object controlling the transition from time 0 to time 1
    '''
    @cached_property
    def _get_sim_step(self):
        return SimulationStep(forming_task=self,
                              config=self.sim_config)

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
