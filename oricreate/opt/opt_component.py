'''
Created on Nov 14, 2014

@author: rch
'''

from traits.api import \
    HasStrictTraits, WeakRef, Property


class OptComponent(HasStrictTraits):

    '''Component of the optimization problem.
    Base class for goal functions, equality constraints, and inequality
    constraints.
    '''

    forming_task = WeakRef

    formed_object = Property

    def _get_formed_object(self):
        return self.forming_task.formed_object
