'''
Created on Oct 6, 2014

@author: rch
'''

from traits.api import Interface


class IGu(Interface):

    '''Interface of an equality constraint.
    '''

    def get_G(self, U, t=0):
        '''Return the vector of equality constraint values.
        '''

    def get_G_du(self, U, t=0):
        '''Return the jacobian of equality constraint values.
        '''
