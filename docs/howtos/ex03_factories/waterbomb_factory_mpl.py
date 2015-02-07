r'''

Plot the results of single vertex example.

'''

if __name__ == '__main__':
    from docs.howtos.ex03_factories.waterbomb_factory import create_cp
    cp = create_cp()
    # begin
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=False)
    plt.tight_layout()
    # end
