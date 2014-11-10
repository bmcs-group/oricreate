'''
Created on Aug 14, 2014

@author: rch
'''

from traits.api import \
    Array, Property, cached_property, Instance, \
    Str, List

import numpy as np

from oricreate.simulation_step import \
    SimulationStep

from oricreate.forming_tasks import \
    FormingTask


class MapToSurface(FormingTask):

    '''The nodes of the formed are  mapped on an target face, without any
    constraints. This will be done for time_step = 0.001, so theirs only
    a little deformation.

    t_init (float): Time step which is used for the final mapping.
    default = 0.001
    '''

    name = Str('init')

    target_surfaces = List(None)

    mapping_operator = Instance(SimulationStep)

    X_0 = Property()
    '''Array of nodal coordinates.
    '''

    def _get_X_0(self):
        return self.formed_object.X.flatten()

    L = Property()
    '''Array of crease_lines defined by pairs of node numbers.
    '''

    def _get_L(self):
        return self.formed_object.L

    F = Property()
    '''Array of crease facets defined by list of node numbers.
    '''

    def _get_F(self):
        return self.formed_object.F

    # cached initial vector
    _U_0 = Array(float)

    U_0 = Property(Array(float))
    '''Attribute storing the optional user-supplied initial array.
    It is used as the trial vector of unknown displacements
    for the current FormingTask simulation.
    '''

    def _get_U_0(self):
        len_U_0 = len(self._U_0)
        if len_U_0 == 0 or len_U_0 != self.n_dofs:
            self._U_0 = np.zeros((self.n_N * self.n_D,), dtype='float_')
        return self._U_0

    def _set_U_0(self, value):
        self._U_0 = value

    U_1 = Property(depends_on='source_config_changed, _U_0, unfold')
    '''Result of the initialization.
    The ``U0`` vector is used as a first choice. If target
    faces for initialization are specified, the
    rigid folding simulator is used to get the projection of the nodes
    to the target surface at the time (t_init).
    '''
    @cached_property
    def _get_U_1(self):
        if self.mapping_operator is None:
            return self.U_0
        else:
            return self.mapping_operator.U_1[-1]

    X_1 = Property(depends_on='source_config_changed, _U_0')
    '''Output of the respaing step.
    '''
    @cached_property
    def _get_X_1(self):
        return self.X_0
