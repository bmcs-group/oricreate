'''
Created on Aug 14, 2014

@author: rch
'''

from traits.api import \
    Array, Property, cached_property, Instance, \
    Float, Int, Str, List

import numpy as np

from oricreate.simulation_step import \
    SimulationStep

from oricreate.crease_pattern import \
    CreasePattern

from oricreate.forming_tasks import \
    FormingTask


class MapToSurface(FormingTask):

    '''Initialization of the pattern for the FormingTask control.
    The crease pattern object will be mapped on an target face, without any
    constraints. This w`ill be done for time_step = 0.001, so theirs only
    a little deformation.

    t_init (float): Time step which is used for the final mapping.
    default = 0.001
    '''

    # ========================================================================
    # Auxiliary elements that can be used interim steps of computation.
    # They are not included in the crease pattern representation.
    # ========================================================================
    x_aux = Array(dtype=float, value=[])
    '''Auxiliary nodes used for visualization.
    '''

    L_aux = Array(dtype=int, value=[])
    '''Auxiliary lines used for visualization.
    '''

    F_aux = Array(dtype=int, value=[])
    '''Auxiliary facets used for visualization.
    '''

    name = Str('init')

    cp = Instance(CreasePattern)
    '''Instance of a crease pattern.
    '''

    target_surfaces = List(None)

    mapping_operator = Instance(SimulationStep)

    X_0 = Property()
    '''Array of nodal coordinates.
    '''

    def _get_X_0(self):
        return self.cp.X.flatten()

    L = Property()
    '''Array of crease_lines defined by pairs of node numbers.
    '''

    def _get_L(self):
        return self.cp.L

    F = Property()
    '''Array of crease facets defined by list of node numbers.
    '''

    def _get_F(self):
        return self.cp.F

    t_init = Float(0.05)
    '''Time step which is used for the initialization mapping.
    '''

    def _t_init_changed(self):
        self.t_arr = np.linspace(0, self.t_init, self.n_steps + 1)

    n_steps = Int(1, auto_set=False, enter_set=True)
    '''Number of time steps for the FormingTask simulation.
    '''

    def _n_steps_changed(self):
        self.t_arr = np.linspace(0, self.t_init, self.n_steps + 1)

    t_arr = Array(float)
    '''Time array to the only given time step which is t_init.
    '''

    def _t_arr_default(self):
        return np.linspace(0, self.t_init, self.n_steps + 1)

    t_init = Float(0.01, auto_set=False, enter_set=True)
    '''Factor defining the fraction of the target face z-coordinate
    moving the nodes in the direction of the face.
    '''

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
