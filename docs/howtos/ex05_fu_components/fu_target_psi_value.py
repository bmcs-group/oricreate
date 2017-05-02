r'''
Example showing how to introduce constraints on individual nodes.
The displacement of the node 0 is prescribed both in x and y direction.

'''
from oricreate.api import CreasePatternState, CustomCPFactory


def create_hu():
    cp = CreasePatternState(X=[[0, 0.5, 0],
                               [1, 0, 0],
                               [1, 1, 0],
                               [2, 0.5, 0],
                               ],
                            L=[[0, 1], [1, 2], [2, 0], [1, 3],
                               [2, 3]],
                            F=[[0, 1, 2], [1, 3, 2]]
                            )

    cp_factory = CustomCPFactory(formed_object=cp)

    # begin
    from oricreate.fu import FuTargetPsiValue
    # Link the crease factory it with the constraint client
    fu_target_psi_value = \
        FuTargetPsiValue(forming_task=cp_factory,
                         psi_value=(1, lambda t: -0.5 * t)
                         )
    cp = cp_factory.formed_object
    print cp.iL
    print cp.iL_psi
    print 'fu:', fu_target_psi_value.get_f(1.0)
    print 'f_du:\n', fu_target_psi_value.get_f_du(1.0)
    # end
    return fu_target_psi_value

if __name__ == '__main__':
    create_hu()
