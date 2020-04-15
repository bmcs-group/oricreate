r'''
This example uses the crease pattern in the constraint on
developability.

The obtained result show that the crease pattern configuration
at hand satisfies the developability condition.
Further, the derivatives of the condition with
respect to the node displacement can be easily verified
by checking if a  node displacement in a respective direction
increases or decreases the angle around the interior node
of the example crease pattern.
'''
from .custom_factory_mpl import create_cp_factory


def create_fu():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.gu import GuDevelopability
    # Link the pattern factory with the constraint client.
    gu_devel = GuDevelopability(forming_task=cp_factory)
    print('gu:', gu_devel.get_G())
    print('g_du:\n', gu_devel.get_G_du())
    # end
    return gu_devel

if __name__ == '__main__':
    create_fu()
