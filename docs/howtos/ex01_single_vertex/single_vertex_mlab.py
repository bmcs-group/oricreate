r'''

Plot the results of single vertex example.

'''

if __name__ == '__main__':
    from single_vertex import create_cp
    cp = create_cp()
    # begin
    from mayavi import mlab
    cp.plot_mlab(mlab)
    mlab.show()
    # end
