if __name__ == '__main__':

    import numpy as np

    from oricreate.api import \
        CreasePattern, CustomCPFactory, \
        MapToSurface, FoldRigidly, FormingView

    from oricreate.fu import \
        FuPotentialEnergy

    cp = CreasePattern(X=[[0, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0],
                          [1, 1, 0],
                          [2, 0, 0],
                          [2, 1, 0],
                          [3, 0, 0],
                          [3, 1, 0]],
                       L=[[0, 1], [0, 2], [2, 3], [1, 3], [0, 3],
                          [2, 3], [2, 4], [4, 5], [3, 5], [2, 5],
                          [4, 5], [4, 6], [6, 7], [5, 7], [4, 7],
                          ],
                       F=[[0, 1, 2], [1, 2, 3],
                          [2, 3, 4], [3, 4, 5],
                          [4, 5, 6], [5, 6, 7]
                          ]
                       )
    cf = CustomCPFactory(formed_object=cp)

    init = MapToSurface(previous_task=cf)
    init.t_arr
    init.u_t[-1]

    fold = FoldRigidly(previous_task=init, n_steps=1,
                       acc=1e-6, MAX_ITER=500,
                       )

    fold.u_t[-1]

    oc = FuPotentialEnergy(forming_task=init)

    u = np.zeros_like(cp.X)
    print 'f', oc.get_f(u)
    print 'f_du', oc.get_f_du(u)

    cpw = FormingView(root=init)
    cpw.configure_traits()
