if __name__ == '__main__':
    from oricreate import \
        CreasePattern, MapToSurface

    from oricreate.opt_crit import \
        OptCritNodeDist

    import numpy as np

    cp = CreasePattern(X=[[0, 0, 0],
                          [0.5, 0, 0],
                          [10.0, 0, 0]],
                       L=[[0, 1], [1, 2], [2, 0]])
    init = MapToSurface(cp=cp)
    oc = OptCritNodeDist(FormingTask=init,
                         L=[[0, 1], [1, 2]])

    u = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]], dtype='f')
    print 'f', oc.get_f(u)
    print 'f_du', oc.get_f_du(u)
