#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 7, 2011 by: rch

from traits.api import \
    Property, cached_property, \
    Array, Constant

import numpy as np

from crease_pattern_export import \
    CreasePatternExport
from crease_pattern_plot_helper import \
    CreasePatternPlotHelper
from crease_pattern_operators import \
    CreaseNodeOperators, CreaseLineOperators, CreaseFacetOperators, \
    CreaseCummulativeOperators

INPUT = '+cp_input'

class CreasePattern(CreaseNodeOperators,
                    CreaseLineOperators,
                    CreaseFacetOperators,
                    CreaseCummulativeOperators,
                    CreasePatternPlotHelper,
                    CreasePatternExport):
    '''
    Representation of a crease pattern.
    The input attributes of a crease pattern are

    * array of node coordinates -- ``X``
    * array of lines -- ``L``
    * array of facets -- ``F``

    All remaining attributes providing topological mappings
    of the crease pattern are provided as property
    attributes calculated on demand using the input values.
    The mappings implemented in this class
    are independent on the interim configuration of the reshaping
    process of the crease pattern.
    Recalculation of derived properties is initiated only upon a change
    of the mentioned input attributes.

    Geometric characteristics, like line lengths, sector angles, dihedral
    angles, surface areas, potential energy and their derivatives
    are implemented in the operator classes.
    Each of the state characterization methods starts with
    ``get_`` followed by the variable name and has
    the parameter ``u`` representing the interim configuration
    of the crease pattern.
    '''

    #===============================================================================
    # Node coordinates
    #===============================================================================

    X = Array(value=[], dtype='float_', cp_input=True)
    '''Input array of node coordinates with rows specifying ``(n_N,n_D)`` values.
    '''

    x_0 = Property(depends_on=INPUT)
    '''Array of initial coordinates ``(n_N,n_D)`` as ``[x1,x2,x3]``.
    Serves as reshaping classes.
    '''
    def _get_x_0(self):
        return self.X
    def _set_x_0(self, x_0):
        self.X = x_0

    x = Property
    '''Array of current coordinates :math:`(n_N, n_D)`.
    '''
    def _get_x(self):
        return self.x_0

    n_D = Constant(3)
    '''Dinensionality of the Euklidian space (constant = 3).
    '''

    #===========================================================================
    # Node mappings
    #===========================================================================

    N = Property(depends_on=INPUT)
    '''Array of all node numbers.
    '''
    @cached_property
    def _get_N(self):
        return np.arange(self.n_N)

    n_N = Property
    '''Number of crease nodes.
    '''
    def _get_n_N(self):
        return self.X.shape[0]

    NN_L = Property(depends_on=INPUT)
    '''Matrix with ``(n_N,n_N)`` entries containing line numbers
    for the connected nodes. For unconnected nodes it contains the value ``-1``
    '''
    @cached_property
    def _get_NN_L(self):
        NN = np.zeros((self.n_N, self.n_N), dtype='int') - 1
        NN[ self.L[:, 0], self.L[:, 1]] = np.arange(self.n_L)
        NN[ self.L[:, 1], self.L[:, 0]] = np.arange(self.n_L)
        return NN

    N_nbr = Property(depends_on=INPUT)
    '''List with ``n_N`` entries, each containing
    an array with neighbor nodes attached to a node :math:`i`.
    '''
    @cached_property
    def _get_N_nbr(self):
        # identify the neighbors by searching for all lines
        # which mention a node n
        rl = [np.where(self.L == n) for n in self.N]
        # from the identified pair get the node other than the
        # outgoing node
        switch_idx = np.array([1, 0], dtype='int')
        return [self.L[row, switch_idx[col]] for row, col in rl]

    iN = Property(depends_on=INPUT)
    '''Array of interior nodes.
    Included are all nodes that have a closed cycle of neighbors.
    For these nodes, the mappings :math:`\mathrm{nbr}(i,\\kappa)` and
    :math:`\mathrm{aln}(i,\\lambda)` (see below) are provided.
    '''
    @cached_property
    def _get_iN(self):
        return self._get_nbr_cycles()[0]

    iN_nbr = Property(depends_on=INPUT)
    '''List of arrays implementing the mapping
    :math:`\mathrm{nbr}(i,\\kappa)` that delivers the global index
    of a :math:`\\kappa`-th neighbor of an interior node :math:`i` in
    a counter-clockwise order.
    '''
    @cached_property
    def _get_iN_nbr(self):
        return self._get_ordered_nbr_cycles()

    iN_aln = Property(depends_on=INPUT)
    '''List of arrays implementing the mapping
    :math:`\mathrm{aln}(i,\\lambda)` that delivers the global index
    of a :math:`\\lambda`-th line attached to an interior node :math:`i` in
    a counter-clockwise order.
    '''
    @cached_property
    def _get_iN_aln(self):
        iN_aln_lst = []
        for i, neighbors in zip(self.iN, self.iN_nbr):
            iN_aln_lst.append(self.NN_L[i, neighbors[:-1]])
        return iN_aln_lst

    eN = Property(depends_on=INPUT)
    '''Array of edge nodes obtained as a complement of interior nodes.
    '''
    @cached_property
    def _get_eN(self):
        eN_bool = np.ones_like(self.N, dtype='bool')
        eN_bool[self.iN] = False
        return np.where(eN_bool)[0]

    #===========================================================================
    # Line mappings
    #===========================================================================

    L = Array(value=[], dtype='int_', cp_input=True)
    '''Array of crease line nodes ``(n_L,2)`` as index-table ``[n1, n2]``.
    '''
    def _L_default(self):
        return np.zeros((0, 2), dtype='int_')

    n_L = Property
    '''Number of crease lines.
    '''
    def _get_n_L(self):
        return self.L.shape[0]

    LLL_F = Property(depends_on=INPUT)
    '''Matrix with ``(n_L,n_L,n_L)`` entries delivering the facet number
    for a triple of lines constituting this facet.
    For unconnected lines it contains the value ``-1``.
    '''
    @cached_property
    def _get_LLL_F(self):
        LLL = np.zeros((self.n_L, self.n_L, self.n_L), dtype='int') - 1
        LLL[ self.F[:, 0], self.F[:, 1], self.F[:, 2]] = np.arange(self.n_F)
        LLL[ self.F[:, 1], self.F[:, 2], self.F[:, 0]] = np.arange(self.n_F)
        LLL[ self.F[:, 2], self.F[:, 0], self.F[:, 1]] = np.arange(self.n_F)
        LLL[ self.F[:, 2], self.F[:, 1], self.F[:, 0]] = np.arange(self.n_F)
        LLL[ self.F[:, 0], self.F[:, 2], self.F[:, 1]] = np.arange(self.n_F)
        LLL[ self.F[:, 1], self.F[:, 0], self.F[:, 2]] = np.arange(self.n_F)
        return LLL

    L_F_map = Property(depends_on=INPUT)
    '''Array associating lines with the adjacent facets.
    Returns two arrays, the first one contains line indices, the
    second one the facet indices that are attached. Note that
    the mapping is provided for all lines including both interior
    and and edge lines. Lines with no facet attached are not included.
    '''
    @cached_property
    def _get_L_F_map(self):

        # search for facets containing the line numbers
        L = np.arange(self.n_L)

        # use broadcasting to identify the matching indexes in both arrays
        L_F_bool = L[np.newaxis, np.newaxis, :] == self.F_L[:, :, np.newaxis]

        # within the facet any of the line numbers can match, merge the axis 1
        L_F_bool = np.any(L_F_bool, axis=1)
        l, f = np.where(L_F_bool.T)

        return l, f

    iL = Property(depends_on=INPUT)
    '''Array of interior lines ``(n_iL,)``.
    Interior lines are detected by checking that there are two
    attached facets.
    '''
    @cached_property
    def _get_iL(self):
        return np.where(np.bincount(self.L_F_map[0]) == 2)[0]

    eL = Property(depends_on=INPUT)
    '''Array of edge lines ``(n_eL,)``.
    Edge lines are associated to one facet only.
    '''
    @cached_property
    def _get_eL(self):
        return np.where(np.bincount(self.L_F_map[0]) == 1)[0]

    iL_F = Property(depends_on=INPUT)
    '''Array of facets associated with interior lines ``(n_L,2)``.
    Mapping from interior lines to two adjacent facets:
    :math:`\mathrm{afa}(l,\\phi), \; \\phi \\in (1,2)`.
    The ordering of facets is given by the counter-clockwise rotation
    around the first of the line :math:`l`.
    '''
    @cached_property
    def _get_iL_F(self):
        # get the line - to -facet mapping
        l, f = self.L_F_map
        # get the lines that have less than two attached facets
        # i.e. they are edge lines or lose bars
        eL = np.bincount(l) != 2
        # get the indexes within the bincount
        eL_vals = np.where(eL)[0]
        # get the indices of edge lines within the original line array
        el_ix = np.digitize(eL_vals, l) - 1
        # construct the mask hiding the edge lines in the original array
        l_map = np.zeros_like(l, dtype=bool)
        l_map[ el_ix ] = True
        # use the masked array to filter out the edge nodes and lose
        # bars from the mapping.
        fm = np.ma.masked_array(f, mask=l_map)
        # make the array compact and reshape it.
        fm_compressed = np.ma.compressed(fm)
        return fm_compressed.reshape(-1, 2)

    #===========================================================================
    # Facet mappings
    #===========================================================================

    F = Array(value=[], dtype='int_', cp_input=True)
    '''Array of facet nodes ``(n_F,3)`` as an index-table ``[n1, n2, n3]``.
    '''
    def _F_default(self):
        return np.zeros((0, 3), dtype='int_')

    n_F = Property
    '''Number of facets.
    '''
    def _get_n_F(self):
        return self.F.shape[0]

    F_L = Property(depends_on=INPUT)
    '''Lines associated with facets.
    Array with the shape ``(n_F, 3)`` associating each facet with three ``[l1,l2,l3]``.
    '''
    @cached_property
    def _get_F_L(self):
        # cycle indexes around the nodes of a facet
        ix_arr = np.array([[0, 1], [1, 2], [2, 0]])
        # get cycled  node numbers around a facet
        F_N = self.F_N[:, ix_arr]
        # use the NN_L map to get line numbers
        return self.NN_L[F_N[..., 0], F_N[..., 1]]

    n_dofs = Property(depends_on=INPUT)
    '''Total number of displacement degrees of freedom.
    @todo: rename - n_u_dofs
    '''
    @cached_property
    def _get_n_dofs(self):
        return self.n_N * self.n_D

    def _get_nbr_cycle(self, neighbors):
        '''Auxiliary private methods identifying cycles around a node.
        '''
        n_neighbors = len(neighbors)
        neighbor_mtx = self.NN_L[ np.ix_(neighbors, neighbors) ]

        neighbor_map = np.where(neighbor_mtx > -1)[1]

        if n_neighbors == 0 or len(neighbor_map) != 2 * n_neighbors:
            return np.array([], dtype='i')

        cycle_map = neighbor_map.reshape(n_neighbors, 2)

        prev_idx = 0
        next_idx1, next_idx = cycle_map[prev_idx]

        cycle = [0]
        for neigh in range(n_neighbors):
            next_row = cycle_map[next_idx]
            cycle.append(next_idx)
            prev_2idx = next_idx
            next_idx = next_row[ np.where(next_row != prev_idx)[0][0] ]
            prev_idx = prev_2idx

        return neighbors[ np.array(cycle) ]

    def _get_nbr_cycles(self):
        connectivity = []
        iN_lst = []
        for i, neighbors in enumerate(self.N_nbr):
            cycle = self._get_nbr_cycle(neighbors)
            if len(cycle):
                connectivity.append((np.array(cycle)))
                iN_lst.append(i)
        return np.array(iN_lst), connectivity

    def _get_ordered_nbr_cycles(self):
        '''List of nodes having cycle of neighbors the format of the list is
        ``[ (node, np.array([neighbor_node1, neighbor_node2, ... neighbor_node1)), ... ]``
        '''
        oc = []
        iN, iN_nbr = self._get_nbr_cycles()
        for n, neighbors in zip(iN, iN_nbr):
            n1, n2 = neighbors[0], neighbors[1]
            v1 = self.X[n1] - self.X[n]
            v2 = self.X[n2] - self.X[n]
            vcross = np.cross(v1, v2)
            if vcross[2] < 0:
                neighbors = neighbors[::-1]
            oc.append(neighbors)
        return oc

if __name__ == '__main__':

    # trivial example with a single triangle positioned

    cp = CreasePattern(X=[[ 0, 0, 0 ],
                          [ 1, 0, 0 ],
                          [ 1, 1, 0],
                          [0.667, 0.333, 0],
                          [0.1, 0.05, 0]],
                       L=[[ 0, 1 ],
                          [ 1, 2 ],
                          [ 2, 0 ]],
                       F=[[0, 1, 2 ]]
                       )

    print 'vectors\n', cp.L_vectors
    print 'lengths\n', cp.L_lengths
