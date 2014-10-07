r'''

This example demonstrates the interface of a CreasePattern class
using the depicted node-line-facet structure.

.. image:: ex01_single_vertex.png

'''

def create_cp():
    # begin
    from oricreate import CreasePattern

    cp = CreasePattern(X=[[0, 0, 0], [3, 0, 0], [0, 1, 0], [3, 1, 0],
                          [0.5, 0.5, 0]],
                       L=[[0, 1], [0, 2], [1, 3], [2, 3],
                          [0, 4], [1, 4], [3, 4], [2, 4]],
                       F=[[0, 1, 4], [1, 3, 4], [3, 2, 4], [0, 2, 4]])

    print 'Nodes of facets enumerated counter clock-wise\n', cp.F_N
    print 'Lines of facets enumerated counter clock-wise\n', cp.F_L
    # end
    return cp

if __name__ == '__main__':
    create_cp()