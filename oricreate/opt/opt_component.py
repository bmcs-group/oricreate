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

    def validate_input(self):
        '''Method to validate the consistency of the input parameters 
        '''
        return

    def _get_formed_object(self):
        return self.forming_task.formed_object

    U = Property

    def _get_U(self):
        return self.formed_object.U

    u = Property

    def _get_u(self):
        return self.formed_object.u

    X = Property

    def _get_X(self):
        return self.formed_object.X

    x = Property

    def _get_x(self):
        return self.formed_object.x
