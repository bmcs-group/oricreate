r'''
This example uses the crease pattern in the constraint on
constant length or rigidity of facets.

In order to show a non-zero value of the constraint residue,
node displacement in vertical direction is assigned
to the interior node of the sample crease pattern.
'''
from .custom_factory_mpl import create_cp_factory


def create_gu():
    cp_factory = create_cp_factory()
    # begin
    from oricreate.gu import GuConstantLength
    # Link the crease factory it with the constraint client
    gu_constant_length = GuConstantLength(forming_task=cp_factory)
    cp = cp_factory.formed_object
    # Set a vertical of the mid node
    cp.u[4, 2] = -1.0
    cp.u = cp.u
    print('gu:', gu_constant_length.get_G(cp.u))
    print('g_du:\n', gu_constant_length.get_G_du(cp.u))
    # end
    return gu_constant_length

if __name__ == '__main__':
    create_gu()
