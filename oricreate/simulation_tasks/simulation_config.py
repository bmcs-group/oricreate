'''
Created on Oct 29, 2014

@author: rch
'''

from traits.api import \
    HasStrictTraits, Property, cached_property, \
    Int, Trait, DelegatesTo, WeakRef, Dict, Str, Array, List

from opt_crit import \
    OptCritTargetFaces, OptCritPotentialEnergy

from eq_cons import \
    IEqCons

import numpy as np


class SimulationConfig(HasStrictTraits):

    '''Configuration of the optimization problem
    including the goal functions, and constraints.
    '''
    debug_level = Int(0, label='Debug level', auto_set=False, enter_set=True)

    goal_function_type = Trait('target_faces',
                               {'none': None,
                                'target_faces': OptCritTargetFaces,
                                'potential_energy': OptCritPotentialEnergy
                                },
                               input_change=True)
    '''Type of the sampling of the random domain
    '''

    sim_step = WeakRef

    goal_function = Property(depends_on='+input_change')

    @cached_property
    def _get_goal_function(self):
        if self.goal_function_type_:
            return self.goal_function_type_(simulation_step=self.sim_step)
        else:
            return None

    tf_lst = DelegatesTo('goal_function')
    '''List of target faces.

    If target face is available, than use it for initialization.
    The z component of the face is multiplied with a small init_factor
    '''

    # ==========================================================================
    # Equality constraints
    # ==========================================================================

    eqcons = Dict(Str, IEqCons)
    '''Equality constraints
    '''

    def _eqcons_default(self):
        return {}

    eqcons_lst = Property(depends_on='eqcons')

    @cached_property
    def _get_eqcons_lst(self):
        return self.eqcons.values()

    # ===========================================================================
    # Kinematic constraints
    # ===========================================================================

    CS = Array()
    '''Control Surfaces.
    '''

    def _CS_default(self):
        return np.zeros((0,))

    dof_constraints = Array
    '''List of explicit constraints specified as a linear equation.
    '''

    # =======================================================================
    # Constraint data
    # =======================================================================

    GP = List([])
    ''''Points for facet grabbing [node, facet].
        First index gives the node, second the facet.
    '''
    n_GP = Property
    ''''Number of grab points.
    '''

    def _get_n_GP(self):
        '''Number of grab points'''
        return len(self.GP)

    LP = List([])
    '''Nodes movable only on a crease line Array[node,line].
       first index gives the node, second the crease line.
    '''

    n_LP = Property
    '''Number of line points.
    '''

    def _get_n_LP(self):
        return len(self.LP)

    cf_lst = List([])
    '''List of sticky faces defined as a list of tuples
    with the first entry defining the face geometry depending
    on time parameter and second entry specifying the nodes
    sticking to the surface.
    '''

    ff_lst = Property
    '''Derived list of sticky faces without the associated nodes.
    '''

    def _get_ff_lst(self):
        return [ff for ff, nodes in self.cf_lst]  # @UnusedVariable

    n_c_ff = Property
    '''Number of sticky faces.
    '''

    def _get_n_c_ff(self):
        '''Number of constraints'''
        n_c = 0
        # count the nodes in each entry in the cf_lst
        for ff, nodes in self.cf_lst:  # @UnusedVariable
            n_c += len(nodes)
        return n_c
