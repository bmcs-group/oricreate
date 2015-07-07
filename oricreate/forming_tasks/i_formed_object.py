'''
Created on Nov 3, 2014

@author: rch
'''

from traits.api import \
    Interface, Array, HasStrictTraits

import traits.has_traits as ht
ht.CHECK_INTERFACES = 2


class IFormedObject(Interface):

    r'''Interface for geometric object
    to be formed within the forming task.
    '''

    X = Array(dtype='float_')
    L = Array(dtype='int_')
    F = Array(dtype='int_')


class FormedObject(HasStrictTraits):

    r'''Object to be transformed to a new shape.
    '''
