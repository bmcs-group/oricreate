r'''

Plot the results of single vertex example.

'''

if __name__ == '__main__':
    from docs.howtos.ex01_single_vertex.single_vertex import create_cp
    cp = create_cp()
    # begin
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_mpl(ax, facets=True)
    plt.tight_layout()
    # end
