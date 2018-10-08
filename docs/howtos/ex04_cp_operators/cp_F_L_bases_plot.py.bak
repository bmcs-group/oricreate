r'''

Plot the results of single vertex example.

'''

if __name__ == '__main__':
    from docs.howtos.ex04_cp_operators.cp_F_L_bases import \
        create_cp
    from oricreate.crease_pattern import \
        CreasePatternBasesViz3D
    from oricreate.api import FTV

    cp = create_cp()
    viz3d_normals = CreasePatternBasesViz3D(
        label='normals', vis3d=cp)

    ftv = FTV()
    ftv.add(cp.viz3d)
    ftv.add(viz3d_normals)

    m = ftv.mlab
    fig = m.gcf()
    m.figure(figure=fig, bgcolor=(1, 1, 1))
    ftv.plot()

    m.show()
    arr = m.screenshot()
    import pylab as p
    p.imshow(arr)
    p.axis('off')
    p.show()
