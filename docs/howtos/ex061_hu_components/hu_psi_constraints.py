r'''
Example showing how to introduce constraints on individual nodes.
The displacement of the node 0 is prescribed both in x and y direction.

'''
from oricreate.api import CreasePatternState, CustomCPFactory


def create_hu():
    cp = CreasePatternState(X=[[0, 0.5, -0.5],
                               [1, 0, 0],
                               [1, 1, 0],
                               [2, 0.5, -0.5],
                               ],
                            L=[[0, 1], [1, 2], [2, 0], [1, 3],
                               [2, 3]],
                            F=[[0, 1, 2], [1, 3, 2]]
                            )

    cp_factory = CustomCPFactory(formed_object=cp)

    # begin
    from oricreate.hu import HuPsiConstraints
    # Link the crease factory it with the constraint client
    hu_psi_constraints = \
        HuPsiConstraints(forming_task=cp_factory,
                         psi_constraints=[(1, False), ]
                         )
    cp = cp_factory.formed_object
    print cp.iL
    print cp.iL_psi
    print 'gu:', hu_psi_constraints.get_H(1.0)
    print 'g_du:\n', hu_psi_constraints.get_H_du(1.0)
    # end
    return hu_psi_constraints

if __name__ == '__main__':
    create_hu()
