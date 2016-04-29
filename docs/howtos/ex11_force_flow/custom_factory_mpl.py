r'''

Construct a general crease pattern configuration,
used in the examples below demonstrating the evaluation of goal functions
and constraints.
'''


def create_cp_factory():
    # begin
    from oricreate.api import CreasePatternState, CustomCPFactory

    cp = CreasePatternState(X=[[0, 0, 0],
                               [1, 0, 0],
                               [1, 1, 0],
                               [2, 1, 0]
                               ],
                            L=[[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]],
                            F=[[0, 1, 2], [1, 3, 2]]
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
