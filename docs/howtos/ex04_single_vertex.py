r'''

This example demonstrates the interface of a CreasePattern class
using the depicted node-line-facet structure.

.. image:: figs/ex04_single_vertex.png

'''


if __name__ == '__main__':

    # end_doc

    from oricreate import \
        CreasePattern

    cp = CreasePattern(X=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                          [0.5, 0.5, 0]],
                       L=[[0, 1], [0, 2], [1, 3], [2, 3],
                          [0, 4], [1, 4], [3, 4], [2, 4]],
                       F=[[0, 1, 4], [1, 3, 4], [3, 2, 4], [0, 2, 4]])

    print 'Nodes of a facet counter-clockwise\n', cp.F_N

    print 'Lines of a facet counter-clockwise\n', cp.F_L
