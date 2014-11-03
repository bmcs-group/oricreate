'''
Created on Oct 6, 2014

@author: rch
'''

from traits.api import \
    Property, Array, Instance

from crease_pattern import \
    CreasePattern

from forming_tasks import \
    IFormingTask  # @UnresolvedImport


class ISimulationTask(IFormingTask):

    '''Interface for FormingTask process
    simulation step within the origami design process.
    '''

    # method required by subsequent FormingTask steps
    U_1 = Property(Array)
    cp = Instance(CreasePattern)

    # method required for visualization
    U_t = Property(Array)
