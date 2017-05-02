'''
Created on Oct 6, 2014

@author: rch
'''

from traits.api import Interface


class IHu(Interface):

    '''Interface of an inequality constraint.
    '''

    def get_H(self, t=0):
        '''Return the vector of equality constraint values.
        '''

    def get_H_du(self, t=0):
        '''Return the jacobian of equality constraint values.
        '''
