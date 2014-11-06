'''
Created on Jun 20, 2013

@author: rch
'''
from traits.api import \
    Property, Array, cached_property

from mapping_task import MappingTask
import numpy as np


class MaskFacets(MappingTask):

    '''Use the source element as a base and define a mask for hidden
    deleted) facets.Nodes and lines without facet will be masked as well
    '''
    name = 'mask'

    F_mask = Array(dtype=int, value=[])

    F = Property(Array, depends_on='F_mask')
    '''Array of crease facets defined by list of node numbers.
    '''
    @cached_property
    def _get_F(self):
        F = self.source.F
        select_arr = np.ones((len(F),), dtype=bool)
        select_arr[self.F_mask] = False
        return F[select_arr]

if __name__ == '__main__':
    from crease_pattern import \
        YoshimuraCPFactory
    yf = YoshimuraCPFactory(n_x=4, n_y=4)
    print yf.formed_object.F
    m = MaskFacets(previous_task=yf, F_mask=[20, 30, 23])
    print yf.formed_object.F

    import pylab as p
    ax = p.axes()
    m.formed_object.plot_mpl(ax)
    p.show()
