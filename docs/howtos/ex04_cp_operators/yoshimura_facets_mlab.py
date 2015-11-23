r'''

Plot the results of single vertex example.

'''

if __name__ == '__main__':
    from docs.howtos.ex04_cp_operators.yoshimura_facets_normals import \
        create_cp
    cp = create_cp()

    cp.u[5, 2] = 1.0
    cp.u[6, 2] = 1.0
    cp.u = cp.u
    # begin
    import mayavi.mlab as m
    m.figure(bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.6, 0.6, 0.6))
    cp.plot_mlab(m, lines=True)

    Fa_r = cp.Fa_r
    Fa_normals = cp.Fa_normals

    x, y, z = Fa_r.reshape(-1, 3).T
    u, v, w = Fa_normals.reshape(-1, 3).T

    m.points3d(x, y, z, scale_factor=0.1)
    m.quiver3d(x, y, z, u, v, w)
    # end
    m.show()
