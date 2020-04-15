'''
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
# Created on Apr 25, 2013

'''

import collections
import types

import numpy as np


def _broadcast_nd(n, d):
    '''Construct the combination of supplied node and dimension indexes.
    '''
    nodes, dirs = n, d
<<<<<<< master
    if isinstance(nodes, np.int_):
        nodes = np.array([n], dtype=np.int_)
=======
    print(type(n))
    print(type(d))
    if isinstance(nodes, (int, np.int_)):
        nodes = np.array([n], dtype='int')
>>>>>>> Transformed to python 3
    elif isinstance(nodes, collections.Container):
        nodes = np.array(list(nodes), np.int_)

<<<<<<< master
    if isinstance(dirs, int):
        dirs = np.array([d], dtype=np.int_)
=======
    if isinstance(dirs, (int, np.int_)):
        dirs = np.array([d], dtype='int')
>>>>>>> Transformed to python 3
    elif isinstance(dirs, collections.Container):
        dirs = np.array(list(dirs), dtype=np.int_)
    return np.broadcast(nodes[None, :], dirs[:, None])


def fix(nodes, dirs, fn=None):
    '''Return constraints corresponding to the specified arrys of nodes and dirs.

    For example, given the call::

        fix([2,4],[0,1])

    defines the dof_constraints of the form:

    [([(2, 0, 1.0)], 0.0, None ), ([(2, 1, 1.0)], 0.0, None ),
     ([(4, 0, 1.0)], 0.0, None), ([(4, 1, 1.0)], 0.0, None ),]

    The structure of the dof_constraints list
    is explained here :class:`DofConstraints`
    '''
    return [([(n, d, 1.0)], fn) for n, d in _broadcast_nd(nodes, dirs)]


def link(nodes1, dirs1, c1, nodes2, dirs2, c2, fn=None):
    '''Return constraints corresponding to the specified arrys of nodes and dirs.

    For example, given the call::

        link([2,4],[0],1.0,[0,1],[0],-1.0)

    defines the dof_constraints of the form:

    [([(2, 0, 1.0), (0, 0, -1.0)], -1.0),
     ([(4, 0, 1.0), (1, 0, -1.0)], -1.0)]

    The structure of the dof_constraints list
    is explained here :class:`DofConstraints`
    '''
    print('BROADCAST')
    print(nodes1)
    print(dirs1)
    bc_dofs1 = np.array([[n, d] for n, d in _broadcast_nd(nodes1, dirs1)])
    print(nodes2)
    print(dirs2)
    bc_dofs2 = np.array([[n, d] for n, d in _broadcast_nd(nodes2, dirs2)])
    bc_linked_dofs = np.broadcast_arrays(bc_dofs1,
                                         bc_dofs2)
    return [([(n1, d1, c1), (n2, d2, c2)], fn)
            for (n1, d1), (n2, d2) in zip(*bc_linked_dofs)]
