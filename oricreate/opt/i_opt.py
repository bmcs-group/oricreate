'''
Created on Oct 29, 2014

@author: rch
'''

from traits.api import \
    Interface, Property, Instance, \
    Dict, Str

from .i_fu import \
    IFu
from .i_gu import \
    IGu
from .i_hu import \
    IHu


class IOpt(Interface):

    r'''Configuration of the optimization problem
    including the goal functions, and constraints.

    .. math::
       \min f(\bm{u}), \; \bm{g}(\bm{u}) = 0, \; \bm{h}(\bm{h}) > 0

    '''
    fu = Instance(IFu)
    '''Goal function of the optimization problem.
    '''

    gu = Dict(Str, IGu)
    '''Equality constraints
    '''

    def _gu_default(self):
        return {}

    gu_lst = Property
    '''Ordered list of equality constraints
    '''
    hu = Dict(Str, IHu)
    '''Inequality constraints
    '''

    def _hu_default(self):
        return {}

    hu_lst = Property
    '''Ordered list of inequality constraints
    '''
