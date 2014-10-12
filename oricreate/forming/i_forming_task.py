'''
Created on Oct 6, 2014

@author: rch
'''

from traits.api import Property, Array, Instance
from oricreate import CreasePattern
from i_ori_node import IOriNode

class IFormingTask(IOriNode):
    '''Interface for FormingTask process
    simulation step within the origami design process.
    '''

    # method required by subsequent FormingTask steps
    U_1 = Property(Array)
    cp = Instance(CreasePattern)

    # method required for visualization
    U_t = Property(Array)
