r'''

Plot the results of single vertex example.

'''


def create_cp_factory():
    # begin
    from oricreate.api import YoshimuraCPFactory
    cp_factory = YoshimuraCPFactory(n_x=1, n_y=2, L_x=2, L_y=1)
    # end
    return cp_factory

if __name__ == '__main__':
    cp_factory = create_cp_factory()
    cp = cp_factory.formed_object
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=False)
    plt.tight_layout()
