r'''
Example showing how to introduce constraints on individual nodes.
The displacement of the node 0 is prescribed both in x and y direction.

'''
from custom_factory_mpl import create_cp_factory


def create_gu():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.gu import GuDofConstraints
    # Link the crease factory it with the constraint client
    gu_dof_constraints = \
        GuDofConstraints(forming_task=cp_factory,
                         dof_constraints=[([(0, 0, 1.0)], 0.1),
                                          ([(0, 1, 1.0)], 0.2),
                                          ])
    print 'gu:', gu_dof_constraints.get_G(1.0)
    print 'g_du:\n', gu_dof_constraints.get_G_du(1.0)
    # end
    return gu_dof_constraints

if __name__ == '__main__':
    create_gu()
