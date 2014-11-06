'''
Created on Nov 6, 2014

@author: rch
'''
from traits.api import \
    Interface


class IOptCrit(Interface):

    '''Interface of an equality constraint.
    '''

    def get_f(self, u, t=0):
        '''Return the vector of equality constraint values.
        '''

    def get_f_du(self, u, t=0):
        '''Return the Jacobian of equality constraint values.
        '''
