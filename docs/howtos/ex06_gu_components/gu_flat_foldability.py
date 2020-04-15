r'''
This example uses the crease pattern in the constraint on
flat-foldability.

The obtained result show that the crease pattern configuration
at hand satisfies the flat-foldability condition.
Further, the derivatives of the condition with
respect to the node displacement are all zero.
This means in this case, that the displacement of any node
in any direction affects/violates
the condition with the same rate.
'''
from .custom_factory_mpl import create_cp_factory


def create_fu():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.gu import GuFlatFoldability
    # Link the pattern factory with the constraint client.
    gu_devel = GuFlatFoldability(forming_task=cp_factory)
    print('gu:', gu_devel.get_G())
    print('g_du:\n', gu_devel.get_G_du())
    # end
    return gu_devel

if __name__ == '__main__':
    create_fu()
