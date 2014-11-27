'''
Created on Nov 14, 2014

@author: rch
'''

from traits.api import \
    HasStrictTraits, WeakRef


class OptComponent(HasStrictTraits):

    '''Component of the optimization problem.
    Base class for goal functions, equality constraints, and inequality
    constraints.
    '''

    forming_task = WeakRef
