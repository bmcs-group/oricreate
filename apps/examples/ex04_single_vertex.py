r'''

Given are four nodes and two pairs of vectors

.. image:: figs/ex04_single_vertex.png

The triangulation can be constructed as::

    cp = CreasePattern(X=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],[0.5, 0.5, 0]],
                       L=[[0, 1], [0, 2], [1, 3], [2, 3],
                          [0, 4], [1, 4], [3, 4], [2, 4]],
                       F=[[0, 1, 4], [1, 3, 4], [3, 2, 4], [0, 2, 4]])

the operators provided by this class include

Mapping between facets and nodes::

    In ]1]: print(cp.F_N)
    [[0 1 4]
     [1 3 4]
     [3 2 4]
     [4 2 0]]

Note that the nodes are enumerated counter-clockwise.
The mapping between facets enumerated counter clockwise
is obtained using::

    In [2[: print(cp.F_L)
    [[0 5 4]
     [2 6 5]
     [3 7 6]
     [7 1 4]]

'''

if __name__ == '__main__':

    from oricreate import \
        CreasePattern

    cp = CreasePattern(X=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                          [0.5, 0.5, 0]],
                       L=[[0, 1], [0, 2], [1, 3], [2, 3],
                          [0, 4], [1, 4], [3, 4], [2, 4]],
                       F=[[0, 1, 4], [1, 3, 4], [3, 2, 4], [0, 2, 4]])

    print(cp.F_N)

    print(cp.F_L)