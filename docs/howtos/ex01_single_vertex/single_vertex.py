r'''

Let us construct a single vertex crease pattern.
By accessing its property attribute, evaluation of characteristics
like mappings between facets and nodes or facets and lines
are calculated. Further, quantifiable properties like line
vectors or facet areas or normals can be calculated including their
derivatives with respect to the node displacements.

.. image:: single_vertex.png

'''


def create_cp():
    # begin
    from oricreate.api import CreasePatternState

    cp = CreasePatternState(X=[[0, 0, 0], [3, 0, 0], [0, 1, 0], [3, 1, 0],
                               [0.5, 0.5, 0]],
                            L=[[0, 1], [0, 2], [1, 3], [2, 3],
                               [0, 4], [1, 4], [3, 4], [2, 4]],
                            F=[[0, 1, 4], [1, 3, 4], [3, 2, 4], [0, 2, 4]])

    print 'Nodes of facets enumerated counter clock-wise\n', cp.F_N
    print 'Lines of facets enumerated counter clock-wise\n', cp.F_L
    print 'Facet areas\n', cp.F_area
    print 'Vector normal to a face\nt', cp.F_normals
    # end
    return cp

if __name__ == '__main__':
    create_cp()
