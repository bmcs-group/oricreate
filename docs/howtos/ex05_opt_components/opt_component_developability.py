r'''
This script demonstrates the use of a factory
with a client - i.e. a goal function using
the factory product (crease pattern) for evaluation
of the potential energy.
'''
from custom_factory_mpl import create_cp_factory


def create_fu():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.gu import GuDevelopability
    # Link the pattern factory with the constraint client.
    gu_devel = GuDevelopability(cp_factory)
    cp = cp_factory.formed_object
    print 'gu:', gu_devel.get_G(cp.u)
    print 'g_du:\n', gu_devel.get_G_du(cp.u)
    # end
    return gu_devel

if __name__ == '__main__':
    create_fu()
