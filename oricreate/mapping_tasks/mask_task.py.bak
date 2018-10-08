'''
Created on Jun 20, 2013

@author: rch
'''
from traits.api import \
    Array
from mapping_task import \
    MappingTask
import numpy as np
from oricreate.crease_pattern.crease_pattern_state import CreasePatternState


class MaskTask(MappingTask):

    '''Use the source element as a base and define a mask for hidden
    deleted) facets.Nodes and lines without facet will be masked as well
    '''
    name = 'mask'

    N_mask = Array(dtype=int, value=[])
    L_mask = Array(dtype=int, value=[])
    F_mask = Array(dtype=int, value=[])

    def _get_formed_object(self):
        '''Construct a new crease pattern state without masked entities.
        '''
        cp = self.previous_task.formed_object
        x_0, L, F = cp.x_0, cp.L, cp.F

        N_enum = np.arange(len(x_0))

        N_select = np.ones((len(x_0),), dtype=bool)
        N_select[self.N_mask] = False

        N_back_ref = N_enum[N_select]
        N_back_enum = np.arange(len(x_0))

        N_remap = -1 * np.ones((len(x_0),), dtype=bool)
        N_remap[N_back_ref] = N_back_enum

        L_select = np.ones((len(L),), dtype=bool)
        L_select[self.L_mask] = False
        L_reduce = L[L_select]
        L_remapped = N_remap[L_reduce]

        F_select = np.ones((len(F),), dtype=bool)
        F_select[self.F_mask] = False
        F_reduce = F[F_select]
        F_remapped = N_remap[F_reduce]

        return CreasePatternState(x_0=x_0[N_select],
                                  L=L_remapped,
                                  F=F_remapped)


if __name__ == '__main__':
    from oricreate.factories import \
        YoshimuraCPFactory
    yf = YoshimuraCPFactory(n_x=2, n_y=2)
    m = MaskTask(previous_task=yf,
                 N_mask=[7], L_mask=[5, 7, 17], F_mask=[4, 9])
    print yf.formed_object.F

    import pylab as p
    ax = p.axes()
    m.formed_object.plot_mpl(ax)
    p.show()
