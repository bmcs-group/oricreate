'''
Created on Oct 29, 2014

@author: rch
'''

from traits.api import \
    HasStrictTraits, Property, cached_property, provides, \
    Int, Trait, Instance, Bool, Dict, Str,  Float
from traitsui.api import \
    View, UItem,  TableEditor, ObjectColumn, \
    Tabbed, VSplit
from oricreate.fu import \
    FuTargetFaces, FuPotEngGravity, FuPotEngBending, FuPotEngTotal
from oricreate.opt import \
    IOpt, IFu, IGu, IHu

gu_list_editor = TableEditor(
    columns=[ObjectColumn(label='Type', name='label'),
             ],
    editable=False,
    selected='object.selected_gu',
)

hu_list_editor = TableEditor(
    columns=[ObjectColumn(label='Type', name='label'),
             ],
    editable=False,
    selected='object.selected_hu',
)


@provides(IOpt)
class SimulationConfig(HasStrictTraits):

    '''Configuration of the optimization problem
    including the goal functions, and constraints.
    '''

    debug_level = Int(0, label='Debug level', auto_set=False, enter_set=True)
    r'''Debug level for simulation scheme.
    '''

    goal_function_type = Trait('target_faces',
                               {'none': None,
                                'target_faces': FuTargetFaces,
                                'bending potential energy': FuPotEngBending,
                                'gravity potential energy': FuPotEngGravity,
                                'total potential energy': FuPotEngTotal
                                },
                               input_change=True)
    r'''Type of the goal function.
    '''

    _fu = Instance(IFu)
    '''Private trait with the goal function object.
    '''

    fu = Property(Instance(IFu), depends_on='goal_function_type')
    '''Goal function.
    '''

    def _get_fu(self):
        if self._fu == None:
            self._fu = self.goal_function_type_()
        return self._fu

    def _set_fu(self, value):
        if not value.__class__ is self.goal_function_type_:
            raise TypeError('Goal function has type %s but should be %s' %
                            (value.__class__, self.goal_function_type_))
        self._fu = value

    gu = Dict(Str, IGu)
    '''Dictionary of equality constraints.
    '''

    def _gu_default(self):
        return {}

    gu_lst = Property(depends_on='gu')
    '''List of equality constraints.
    '''
    @cached_property
    def _get_gu_lst(self):
        for name, gu in list(self.gu.items()):
            gu.label = name
        return list(self.gu.values())

    selected_gu = Instance(IGu)

    hu = Dict(Str, IHu)
    '''Inequality constraints
    '''

    def _hu_default(self):
        return {}

    hu_lst = Property(depends_on='hu')
    '''List of inequality constraints.
    '''
    @cached_property
    def _get_hu_lst(self):
        return list(self.hu.values())

    selected_hu = Instance(IHu)

    has_H = Property(Bool)

    def _get_has_H(self):
        return len(self.hu) > 0

    show_iter = Bool(False, auto_set=False, enter_set=True)
    r'''Saves the first 10 iteration steps, so they can be analyzed
    '''

    MAX_ITER = Int(100, auto_set=False, enter_set=True)
    r'''Maximum number of iterations.
    '''

    acc = Float(1e-4, auto_set=False, enter_set=True)
    r'''Required accuracy.
    '''

    use_f_du = Bool(True, auto_set=False, enter_set=True)
    r'''Switch the use of goal function derivatives on.
    '''

    use_G_du = Bool(True, auto_set=False, enter_set=True)
    r'''Switch the use of constraint derivatives on.
    '''

    use_H_du = Bool(True, auto_set=False, enter_set=True)
    r'''Switch the use of constraint derivatives on.
    '''

    def validate_input(self):
        # self.fu.validate_input()
        for gu in self.gu_lst:
            gu.validate_input()
        for hu in self.hu_lst:
            hu.validate_input()

    traits_view = View(
        Tabbed(
            VSplit(
                UItem('gu_lst@', editor=gu_list_editor),
                UItem('selected_gu@'),
                label='Gu'
            ),
            VSplit(
                UItem('hu_lst@', editor=hu_list_editor),
                UItem('selected_hu@'),
                label='Hu'
            ),
        )
    )
