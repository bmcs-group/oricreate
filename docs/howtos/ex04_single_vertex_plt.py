r'''

Plot the results of single vertex example.

'''

if __name__ == '__main__':
    from ex04_single_vertex import create_cp
    cp = create_cp()
    # begin
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cp.plot_2D(ax)
    plt.tight_layout()
    # end