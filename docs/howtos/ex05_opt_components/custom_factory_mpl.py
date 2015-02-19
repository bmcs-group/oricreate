r'''

Construct a general crease pattern configuration,
used in the examples evaluating available goal functions
and constraints.
'''


def create_cp_factory():
    # begin
    from oricreate.api import CreasePatternState, CustomCPFactory

    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [1, 1, 0],
                               [0, 1, 0],
                               [0.5, 0.5, 0]
                               ],
                            L=[[0, 1], [1, 2], [2, 3], [3, 0],
                               [0, 4], [1, 4], [2, 4], [3, 4]],
                            F=[[0, 1, 4], [1, 2, 4], [4, 3, 2], [4, 3, 0]]
                            )

    cp_factory = CustomCPFactory(formed_object=cp)
    # end
    return cp_factory

if __name__ == '__main__':
    cp_factory = create_cp_factory()
    cp = cp_factory.formed_object
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=True)
    plt.tight_layout()
    plt.show()
