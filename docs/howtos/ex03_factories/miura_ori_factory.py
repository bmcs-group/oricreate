r'''

This example demonstrates the crease pattern factory usage
for MiuraOri crease pattern.

This crease pattern contains quadrilateral facets.
A quadrilateral facet is assembled out of two triangles
connected with a no-fold-line.

'''


def create_cp():
    # begin
    from oricreate.api import MiuraOriCPFactory

    cp_factory = MiuraOriCPFactory()
    cp = cp_factory.formed_object

    print 'Nodes of facets enumerated counter clock-wise\n', cp.F_N
    print 'Lines of facets enumerated counter clock-wise\n', cp.F_L
    # end
    return cp

if __name__ == '__main__':
    create_cp()
