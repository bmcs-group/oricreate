r'''

Plot the results of single vertex example.

'''

if __name__ == '__main__':
    from docs.howtos.ex04_cp_operators.yoshimura_facets_normals import \
        create_cp
    from oricreate.crease_pattern import \
        CreasePatternNormalsViz3D
    from oricreate.api import FTV

    cp = create_cp()
    viz3d_normals = CreasePatternNormalsViz3D(
        label='normals', vis3d=cp)

    ftv = FTV()
    ftv.add(cp.viz3d)
    ftv.add(viz3d_normals)

    m = ftv.mlab
    fig = m.gcf()
    m.figure(figure=fig, bgcolor=(1, 1, 1))
    ftv.plot()

    arr = m.screenshot()
    import pylab as p
    p.imshow(arr)
    p.axis('off')
    p.show()
