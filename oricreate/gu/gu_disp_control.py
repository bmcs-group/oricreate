'''
Created on Feb 19, 2015

@author: rch
'''

from traits.api import \
    implements, \
    Array, DelegatesTo, cached_property, Property

from gu import Gu
from gu_disp_control_viz3d import \
    GuDofConstraintsViz3D
import numpy as np
from oricreate.opt import \
    IGu
from oricreate.viz3d import \
    Visual3D


class GuGrabPoints(Gu):

    '''Grab points are included in the nodes attribute of the crease pattern.
    Their position is constrained within a facet using triangle coordinates.
    '''
    implements(IGu)

    n_dofs = DelegatesTo('forming_task')
    N = DelegatesTo('forming_task')
    F = DelegatesTo('forming_task')
    GP = DelegatesTo('forming_task')
    n_GP = DelegatesTo('forming_task')
    n_D = DelegatesTo('forming_task')

    # =========================================================================
    # Grab point specification
    # =========================================================================
    grab_pts_L = Property(Array, depends_on='X, F, GP')

    @cached_property
    def _get_grab_pts_L(self):
        '''Calculates the L vector for the Barycentric coordinates
           Trick: assuming a tetrahedron with fourth point on [ 0, 0, -1],
           if the grab point is chosen correctly
           (lying in the plane of the facet)
           L4 will be 0
        '''
        n = self.x_0
        f = self.F

        x4 = np.array([0, 0, -1])
        L = np.array([])

        for i in self.GP:
            f_i = i[1]  # actual facet index
            T = np.c_[n[f[f_i][0]] - x4, n[f[f_i][1]] - x4]
            T = np.c_[T, n[f[f_i][2]] - x4]
            Tinv = np.linalg.inv(T)

            x = n[i[0]] - x4
            Li = np.dot(Tinv, x)
            L = np.append(L, Li)

        L = L.reshape(-1, 3)  # gives L1,L2,L3 for each grabpoint
        return L

    def get_G(self, t=0):
        ''' Calculate the residuum for constant crease length
        given the fold vector U.
        '''
        return np.zeros(self.n_GP * self.n_D,)

    def get_G_du(self, t=0):
        ''' Calculate the residuum for constant crease length
        given the fold vector dX.

        '''
        grab_lines = np.zeros((self.n_GP * self.n_D, self.n_dofs))
        for i in range(len(self.GP)):
            facet = self.F[self.GP[i][1]]
            c = 0
            for q in facet:
                grab_lines[i * 3, q * 3] = self.grab_pts_L[i][c]
                grab_lines[i * 3 + 1, q * 3 + 1] = self.grab_pts_L[i][c]
                grab_lines[i * 3 + 2, q * 3 + 2] = self.grab_pts_L[i][c]
                c += 1

            grab_lines[i * 3, self.GP[i][0] * 3] = -1
            grab_lines[i * 3 + 1, self.GP[i][0] * 3 + 1] = -1
            grab_lines[i * 3 + 2, self.GP[i][0] * 3 + 2] = -1

        return grab_lines


class GuPointsOnLine(Gu):

    '''PointsOnLine are included in the nodes attribute
    of the crease pattern.
    Their position is constrained within a crease
    line-element and at least one other
    constraining Element.
    '''
    implements(IGu)

    LP = DelegatesTo('forming_task')
    n_LP = DelegatesTo('forming_task')
    n_N = DelegatesTo('forming_task')
    n_D = DelegatesTo('forming_task')
    n_dofs = DelegatesTo('forming_task')
    L = DelegatesTo('forming_task')
    N = DelegatesTo('forming_task')

    def get_G(self, t=0):

        line = np.array(self.LP)
        if(len(line) == 0):
            return []
        cl = self.L[line[:, 1]]
        u = self.u
        p0 = self.x_0[line[:, 0]]
        p1 = self.x_0[cl[:, 0]]
        p2 = self.x_0[cl[:, 1]]
        dp0 = u[line[:, 0]]
        dp1 = u[cl[:, 0]]
        dp2 = u[cl[:, 1]]

        ri = p1 + dp1
        rj = p2 + dp2
        rij = p0 + dp0

        # parameter free determinant Form of the line
        R = np.cross((rj - ri), (rij - ri))

        # sorting of the residuum for same arangement as G_du
        # ToDo: Redesigne G_du
        Rx = R[:, 1]
        Ry = R[:, 0] * -1
        Rz = R[:, 2] * -1

        R = np.zeros((len(Rx) * 2,))

        # linepoint Elements take only two equations!
        # if line lays in a system axis the representing equation
        # will be zero, so it will be singular
        for i in range(len(Rx)):
            if((p1[i][0] == p2[i][0])and(p1[i][2] == p2[i][2])):
                # check if line lays on x-axis
                R[i * 2] = Ry[i]
                R[i * 2 + 1] = Rz[i]
            elif((p1[i][1] == p2[i][1])and(p1[i][2] == p2[i][2])):
                # check if line lays on y-axis
                R[i * 2] = Rx[i]
                R[i * 2 + 1] = Rz[i]
            else:
                R[i * 2] = Rx[i]
                R[i * 2 + 1] = Ry[i]

        return R.reshape((-1,))

    def get_G_du(self, t=0):
        ''' Calculate the jacobian of the residuum at the instantaneous
        configuration dR
        '''
        line = np.array(self.LP)
        if(len(line) == 0):
            return np.zeros((self.n_LP * 2, self.n_dofs))
        cl = self.L[line[:, 1]]
        u = self.u
        p0 = self.x_0[line[:, 0]]
        p1 = self.x_0[cl[:, 0]]
        p2 = self.x_0[cl[:, 1]]
        dp0 = u[line[:, 0]]
        dp1 = u[cl[:, 0]]
        dp2 = u[cl[:, 1]]
        dR = np.zeros((len(line) * 2, self.n_dofs))

        for i in range(len(line)):
            if((p1[i][0] == p2[i][0])and(p1[i][2] == p2[i][2])):
                dR1 = self.get_line_G_duf2(
                    p0[i], p1[i], p2[i], dp0[i], dp1[i], dp2[i],
                    line[i], cl[i])
                dR2 = self.get_line_G_duf3(
                    p0[i], p1[i], p2[i], dp0[i], dp1[i], dp2[i],
                    line[i], cl[i])
            elif((p1[i][1] == p2[i][1])and(p1[i][2] == p2[i][2])):
                dR1 = self.get_line_G_duf1(
                    p0[i], p1[i], p2[i], dp0[i], dp1[i], dp2[i],
                    line[i], cl[i])
                dR2 = self.get_line_G_duf3(
                    p0[i], p1[i], p2[i], dp0[i], dp1[i], dp2[i],
                    line[i], cl[i])
            else:
                dR1 = self.get_line_G_duf1(
                    p0[i], p1[i], p2[i], dp0[i], dp1[i], dp2[i],
                    line[i], cl[i])
                dR2 = self.get_line_G_duf2(
                    p0[i], p1[i], p2[i], dp0[i], dp1[i], dp2[i],
                    line[i], cl[i])
            dR[i * 2] = dR1
            dR[i * 2 + 1] = dR2

        return dR

    def get_line_G_duf1(self, p0, p1, p2, dp0, dp1, dp2, line, cl):
        dfdx0 = p2[2] + dp2[2] - p1[2] - dp1[2]
        dfdx1 = p0[2] + dp0[2] - p2[2] - dp2[2]
        dfdx2 = p1[2] + dp1[2] - p0[2] - dp0[2]

        dfdz0 = p1[0] + dp1[0] - p2[0] - dp2[0]
        dfdz1 = p2[0] + dp2[0] - p0[0] - dp0[0]
        dfdz2 = p0[0] + dp0[0] - p1[0] - dp1[0]

        dR = np.zeros((1, self.n_dofs))
        dR[0, line[0] * 3] = dfdx0
        dR[0, line[0] * 3 + 2] = dfdz0
        dR[0, cl[0] * 3] = dfdx1
        dR[0, cl[0] * 3 + 2] = dfdz1
        dR[0, cl[1] * 3] = dfdx2
        dR[0, cl[1] * 3 + 2] = dfdz2

        return dR

    def get_line_G_duf2(self, p0, p1, p2, dp0, dp1, dp2, line, cl):
        dfdy0 = p2[2] + dp2[2] - p1[2] - dp1[2]
        dfdy1 = p0[2] + dp0[2] - p2[2] - dp2[2]
        dfdy2 = p1[2] + dp1[2] - p0[2] - dp0[2]

        dfdz0 = p1[1] + dp1[1] - p2[1] - dp2[1]
        dfdz1 = p2[1] + dp2[1] - p0[1] - dp0[1]
        dfdz2 = p0[1] + dp0[1] - p1[1] - dp1[1]

        dR = np.zeros((1, self.n_dofs))

        dR[0, line[0] * 3 + 1] = dfdy0
        dR[0, line[0] * 3 + 2] = dfdz0
        dR[0, cl[0] * 3 + 1] = dfdy1
        dR[0, cl[0] * 3 + 2] = dfdz1
        dR[0, cl[1] * 3 + 1] = dfdy2
        dR[0, cl[1] * 3 + 2] = dfdz2

        return dR

    def get_line_G_duf3(self, p0, p1, p2, dp0, dp1, dp2, line, cl):
        dfdx0 = p2[1] + dp2[1] - p1[1] - dp1[1]
        dfdx1 = p0[1] + dp0[1] - p2[1] - dp2[1]
        dfdx2 = p1[1] + dp1[1] - p0[1] - dp0[1]

        dfdy0 = p1[0] + dp1[0] - p2[0] - dp2[0]
        dfdy1 = p2[0] + dp2[0] - p0[0] - dp0[0]
        dfdy2 = p0[0] + dp0[0] - p1[0] - dp1[0]

        dR = np.zeros((1, self.n_dofs))
        dR[0, line[0] * 3] = dfdx0
        dR[0, line[0] * 3 + 1] = dfdy0
        dR[0, cl[0] * 3] = dfdx1
        dR[0, cl[0] * 3 + 1] = dfdy1
        dR[0, cl[1] * 3] = dfdx2
        dR[0, cl[1] * 3 + 1] = dfdy2

        return dR


class GuDofConstraints(Gu, Visual3D):

    '''Explicit constraints for selected of freedom.
    '''
    implements(IGu)

    dof_constraints = Array
    '''Specification of explicit constraint for particular degrees of freedom.

    dof constraints are specified as a list of equations with values
    to be inserted on the left- and the right-hand-side of the equation system.
    The dof is identified with the number of the node and the direction (0,1,2)
    for x,y,z values::

        [([(node1, direction1, coefficient1), ... ], value1 ),
         ([(node2, direction2, coefficient2), ... ], value2 )
         ... ]

    Convenience constructors for containers of (node, direction pairs)
    are provided for the most usual cases:
    :func:`oricrete.fix` and :func:`oricrete.link`.
    '''

    def __str__(self):
        s = 'Gu: %s - %d\n' % (self.label, len(self.dof_constraints))
        for i, dof_cnstr in enumerate(self.dof_constraints):
            s += '#:%3d;\n' % i
            lhs, rhs = dof_cnstr
            for n, d, c in lhs:  # @UnusedVariable
                s += '\t+ l:%3d; d:%2d; c:%g;\n' % (n, d, c)
            s += '\t= r: %s\n' % str(rhs)
        return s

    def get_G(self, t=0):
        ''' Calculate the residue for given constraint equations
        '''
        cp = self.formed_object
        u = cp.u
        G = np.zeros((len(self.dof_constraints)), dtype='float_')
        for i, dof_cnstr in enumerate(self.dof_constraints):
            lhs, rhs = dof_cnstr
            for n, d, c in lhs:  # @UnusedVariable
                G[i] += c * u[n, d]
            G[i] -= rhs(t) if rhs else 0
        return G

    def get_G_du(self, t=0.0):
        ''' Calculate the residue for given constraint equations
        '''
        cp = self.formed_object
        G_du = np.zeros((len(self.dof_constraints), cp.n_dofs),
                        dtype='float_')
        for i, dof_cnstr in enumerate(self.dof_constraints):
            lhs, rhs = dof_cnstr  # @UnusedVariable
            for n, d, c in lhs:  # @UnusedVariable
                dof = 3 * n + d
                G_du[i, dof] += c

        return G_du

    viz3d_classes = dict(default=GuDofConstraintsViz3D)
