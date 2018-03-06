'''
Created on Nov 6, 2014

@author: rch
'''

from i_forming_task import \
    IFormingTask
import traits.api as tr


class FormingProcess(tr.HasStrictTraits):

    r'''Forming process starts with the factory tasks 
    followed by several forming tasks including simulations, 
    editing and evaluations.
    '''
    factory_task = tr.Instance(IFormingTask)
