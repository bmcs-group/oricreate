r'''
This script demonstrates the use of a factory
with a client - i.e. a goal function using
the factory product (crease pattern) for evaluation
of the potential energy.
'''
from custom_factory_mpl import create_cp_factory


def create_gu():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.gu import GuConstantLength
    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength(cp_factory)
    cp = cp_factory.formed_object
    # Set a vertical of the mid node
    cp.u[4, 2] = -1.0
    cp.u = cp.u
    print 'gu:', gu_constant_length.get_G(cp.u)
    print 'g_du:\n', gu_constant_length.get_G_du(cp.u)
    # end
    return gu_constant_length

if __name__ == '__main__':
    create_gu()
