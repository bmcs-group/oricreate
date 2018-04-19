'''
Created on Dec 3, 2015

@author: rch
'''

from mayavi import mlab
import os.path
from traits.api import \
    HasTraits, Array, Tuple, Property, List,\
    cached_property

import matplotlib.pyplot as pl
import numpy as np
from oricreate.util.eulerangles import mat2euler, rotation_to_vtk
from oricreate.util.transformations import euler_matrix, euler_from_matrix
from oricreate.viz3d import Viz3D


class CreasePatternSelection(HasTraits):
    '''Subset of crease pattern facets, lines and nodes.
    '''
    N_selection = Array(int, selection_changed=True)
    L_selection = Array(int, selection_changed=True)
    F_selection = Array(int, selection_changed=True)

    x_L_F = Property(Tuple, depends_on='+selection_changed')
    '''Geometry with applied selection arrays.
    '''
    #@cached_property

    def _get_x_L_F(self):
        vis3d = self.vis3d
        x, L, F = vis3d.x, vis3d.L, vis3d.F
        if len(self.N_selection):
            x = x[self.N_selection]
        if len(self.L_selection):
            L = L[self.L_selection]
        if len(self.F_selection):
            F = F[self.F_selection]
        return x, L, F

    x = Property()
    '''Reduced set of coordinates.
    '''
    #@cached_property

    def _get_x(self):
        return self.x_L_F[0]

    L = Property()
    '''Reduced set of coordinates.
    '''
    #@cached_property

    def _get_L(self):
        return self.x_L_F[1]

    F = Property()
    '''Reduced set of coordinates.
    '''
    #@cached_property

    def _get_F(self):
        return self.x_L_F[2]


class CreasePatternDecoratorViz3D(CreasePatternSelection, Viz3D):
    '''Visualize the crease Pattern
    '''

    F_ref = Tuple((16, 10, 12, 14))
    '''Selection of facets used for text mapping.
    '''
    N_ref = Tuple((4, 2, 11, 13))

    F_covered = Array(value=[[]], dtype='int_')

    aplus = Array
    atimes = Array
    rotate = Array(value=[[[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]],
                          [[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]],
                          [[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]],
                          [[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]],
                          [[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]],
                          ], dtype='float_')

    loc_position = Array

    def _get_p_o(self, x_ref, base, at, ap, im_center):
        '''Given the reference point, rotation, basis, multiplier and position vector
        return two tuples specifying the position and orientation of the actor.
        '''
        x, y, z = x_ref + np.einsum('ij,i->j', base, im_center)
        az, ay, ax = np.array(
            mat2euler(base), dtype='float_') / 2. / np.pi * 360.0
        rot1 = np.array([ax, ay, az], dtype='float_')
        ax, ay, az = rot1 * at + ap
        return (x, y, z), (ax, ay, az)

    def _is_interior(self, x_ref, base, lp, F_c, im, ap):
        # scale the image into facet coordinates.
        alpha = np.zeros((im.shape[0], im.shape[1]), dtype='bool')
        Lx, Ly = 2.0, 2.0
        nx, ny = im.shape[:2]
        y_alpha, x_alpha = np.mgrid[Ly:0:complex(0, ny), 0:Lx:complex(0, nx)]
        xy_alpha = np.c_[x_alpha.flatten(), y_alpha.flatten()]
        r_im = np.einsum('ji,kj->ik', base[:2, :2], xy_alpha)
        x_aa, y_aa = r_im
        x_aa += x_ref[0]
        y_aa += x_ref[1]
        x_a = x_aa.reshape(x_alpha.shape)
        y_a = y_aa.reshape(y_alpha.shape)

        vis3d = self.vis3d
        for F in F_c:
            x = vis3d.x[vis3d.F_N[F]]
            r = x[:, :2]
            x1, x2, x3 = r[:, 0]
            y1, y2, y3 = r[:, 1]
            detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            a = ((y2 - y3) * (x_a - x3) +
                 (x3 - x2) * (y_a - y3)) / detT
            b = ((y3 - y1) * (x_a - x3) +
                 (x1 - x3) * (y_a - y3)) / detT
            c = 1.0 - a - b
            aa = np.where(a >= 0, True, False)
            bb = np.where(b >= 0, True, False)
            cc = np.where(c >= 0, True, False)
            abc = aa * bb * cc
            alpha = np.logical_or(alpha, abc)

        xp = x_a.flatten()[::100]
        yp = y_a.flatten()[::100]
        zp = a.flatten()[::100]
        self.ftv.mlab.points3d(xp, yp, zp)
        alpha = np.array(alpha, 'int_')

        return alpha * 255


class FacetsWithTextViz3D(CreasePatternDecoratorViz3D):

    labels = List(['geometry', 'material', 'process', 'function'])

    aplus = Array(value=[[0, 0, 0],
                         [180, 180, 0],
                         [0, 0, 0],
                         [180, 180, 0]], dtype='float_')
    atimes = Array(value=[[-1, 1, 1],
                          [1, 1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]], dtype='float_')

    loc_position = Array(value=[[0.2, 0.2, 0.003],
                                [0.2, 0.2, 0.003],
                                [0.2, 0.2, 0.003],
                                [0.2, 0.2, 0.003],
                                [0.2, 0.2, 0.003]], dtype='float_')

    def plot(self):

        bases = self.vis3d.F_L_bases[self.F_ref, 0, ...]
        m = self.ftv.mlab
        x = self.x
        x[:, 0] += 0.01
        x1s = x[self.N_ref, :]

        texts = []

        for x1, b, r, lp, label, at, ap in zip(x1s, bases, self.rotate, self.loc_position,
                                               self.labels,  self.atimes, self.aplus):
            (x, y, z), (ax, ay, az) = self._get_p_o(x1, r, b, lp, at, ap)
            t = m.text3d(x, y, z, label,
                         orient_to_camera=False,
                         orientation=(ax, ay, az), color=(1, 1, 1), scale=0.21)
            texts.append(t)

        self.texts = texts

    def update(self, vot=0.0):
        self.vis3d.vot = vot
        bases = self.vis3d.F_L_bases[self.F_ref, 0, ...]
        x1s = self.x[self.N_ref, :]

        for x1, t, b, r, lp, at, ap in zip(x1s, self.texts, bases, self.rotate,
                                           self.loc_position, self.atimes, self.aplus):
            (x, y, z), (ax, ay, az) = self._get_p_o(x1, r, b, lp, at, ap)
            t.position = (x, y, z)
            t.orientation = (ax, ay, az)


class FacetsWithImageViz3D(CreasePatternDecoratorViz3D):
    '''Crease pattern facets decorated with images.
    '''

    F_ref = Array(value=[0, 2, 4, 6], dtype='int_')

    N_ref = Array(value=[0, 2, 4, 6], dtype='int_')

    x_refs = Property
    '''Get the coordinates of the reference nodes
    '''

    def _get_x_refs(self):
        vis3d = self.vis3d
        return vis3d.x[self.N_ref]

    bases = Property

    def _get_bases(self):
        return self.vis3d.F_L_bases[self.F_ref, 0, ...]

    F_covered = Array(value=[[0, 9],
                             [2, 11],
                             [4, 13],
                             [6, 15]], dtype='int_')
    atimes = Array(value=[0], dtype='float_')

    im_files = List(
        ['eft01.png', 'eft01.png', 'eft01.png', 'eft01.png'], input=True)

    im_widths = Array(value=[2, 2, 2, 2], dtype='float_')
    '''Length of the horizontal dimension of the image in figure coordinates.
    '''
    im_offsets = Array(value=[[1, 1, 0.005], ], dtype='float_')

    im_arrays = Property(List, depends_on='+input')
    '''Image arrays read into list.
    '''
    @cached_property
    def _get_im_arrays(self):
        upath = os.path.expanduser('~')
        return [pl.imread(os.path.join(upath, 'Downloads', im_file), format='png') * 255
                for im_file in self.im_files]

    im_shapes = Property(Array, depends_on='+input')
    '''Shapes of arrays.
    '''
    @cached_property
    def _get_im_shapes(self):
        return np.array([im.shape for im in self.im_arrays])

    im_scales = Property(Array, depends_on='+input')
    '''Scale of images derived using the width'''
    @cached_property
    def _get_im_scales(self):
        return self.im_widths / self.im_shapes[:, 1]

    im_centers = Property(Array, depends_on='+input')
    '''Get the local image centers.'''
    @cached_property
    def _get_im_centers(self):
        z0 = np.zeros_like(self.im_widths)
        c0 = np.c_[self.im_widths / 2.0,
                   self.im_scales * self.im_shapes[:, 0] / 2.0, z0]
        return c0 + self.im_offsets

    glb_centers = Property(Array, depends_on='vis3d_changed, +input')
    '''Centers of images in global coordinates'''
    @cached_property
    def _get_glb_centers(self):
        return self.x_refs + np.einsum('...ij,...i->...j', self.bases, self.im_centers)

    rotated_bases = Property(Array, depends_on='vis3d_changed, +input')
    '''Rotate the base within the facet plane.
    '''
    @cached_property
    def _get_rotated_bases(self):
        at = self.atimes * np.pi / 180.0
        cat, sat = np.cos(at), np.sin(at)
        zer, one = np.zeros_like(cat), np.ones_like(cat)
        r2d = np.array([[cat, -sat, zer], [sat, cat, zer], [zer, zer, one]])
        r2d = np.swapaxes(r2d, 0, 2)
        r2d = np.swapaxes(r2d, 1, 2)
        rb = np.einsum('...ij,...jk->...ik', r2d, self.bases)
        return rb

    glb_euler_angles = Property(Array, depends_on='vis3d_changed, +input')
    '''Get the euler angles.
    '''
    @cached_property
    def _get_glb_euler_angles(self):
        ea = np.array([euler_from_matrix(base.T, 'syxz')
                       for base in self.rotated_bases], dtype='float_') / np.pi * 180.0
        return ea[:, (1, 0, 2)]

    alpha_channels = Property(Array, depends_on='vis3d_changed, +input')
    '''Get the alpha chanel making the pixels outside the facet transparent.
    '''
    @cached_property
    def _get_alpha_channels(self):
        achannel = []
        for im_array, im_shape, c, s, rb, F_c in zip(self.im_arrays,
                                                     self.im_shapes,
                                                     self.glb_centers,
                                                     self.im_scales,
                                                     self.rotated_bases,
                                                     self.F_covered):
            # scale and rotate the image within the plane coordinates
            nx, ny = im_shape[:2]
            Lx, Ly = nx * s, ny * s
            alpha = np.zeros((nx, ny), dtype='bool')
            x_alpha, y_alpha = np.mgrid[
                0:Lx:complex(0, nx), 0:Ly:complex(0, ny)]
            x_alpha -= Lx / 2.0
            y_alpha -= Ly / 2.0
            xy_alpha = np.c_[x_alpha.flatten(), y_alpha.flatten()]
            r_im = np.einsum('ji,kj->ik', rb[:2, :2], xy_alpha)
            x_aa, y_aa = r_im
            x_aa += c[0]
            y_aa += c[1]
            x_a = x_aa.reshape(x_alpha.shape)
            y_a = y_aa.reshape(y_alpha.shape)

            # get the barycentric coordinates of the facet
            vis3d = self.vis3d
            for F in F_c:
                x = vis3d.x[vis3d.F_N[F]]
                r = x[:, :2]
                x1, x2, x3 = r[:, 0]
                y1, y2, y3 = r[:, 1]
                detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                a = ((y2 - y3) * (x_a - x3) +
                     (x3 - x2) * (y_a - y3)) / detT
                b = ((y3 - y1) * (x_a - x3) +
                     (x1 - x3) * (y_a - y3)) / detT
                c = 1.0 - a - b
                aa = np.where(a >= 0, True, False)
                bb = np.where(b >= 0, True, False)
                cc = np.where(c >= 0, True, False)
                abc = aa * bb * cc
                alpha = np.logical_or(alpha, abc)

            alpha = np.array(alpha, 'int_')
            orig_alpha = im_array[:, :, -1]
            a = np.c_[alpha.flatten(), orig_alpha.flatten()]
            aa = np.min(a, axis=1)
            achannel.append(aa * 255)
        return achannel

    def plot(self):
        """
        Test if mlab_imshowColor displays correctly by plotting the wikipedia png example image
        """

        imshows = []
        for im_array, p, o, s, a in zip(self.im_arrays, self.glb_centers,
                                        self.glb_euler_angles, self.im_scales,
                                        self.alpha_channels):
            imshow = self._plot_imshow(im_array[:, :, :3], a,
                                       orientation=o, position=p, scale=s)
            imshows.append(imshow)
        self.imshows = imshows

    def update(self):
        '''Update the orientation and position of the image
        '''
        for imshow, p, o in zip(self.imshows, self.glb_centers,
                                self.glb_euler_angles):
            imshow.actor.position = p
            imshow.actor.orientation = o

    def _plot_imshow(self, im, alpha,
                     position=(0, 0, 0),
                     orientation=(0, 0, 0),
                     scale=1.0, **kwargs):
        """
        Plot a color image with mayavi.mlab.imshow.
        im is a ndarray with dim (n, m, 3) and scale (0->255]
        alpha is a single number or a ndarray with dim (n*m) and scale (0->255]
        **kwargs is passed onto mayavi.mlab.imshow(..., **kwargs)
        """
        if len(alpha.shape) != 1:
            alpha = alpha.flatten()

        # The lut is a Nx4 array, with the columns representing RGBA
        # (red, green, blue, alpha) coded with integers going from 0 to 255,
        # we create it by stacking all the pixles (r,g,b,alpha) as rows.
        lut = np.c_[im.reshape(-1, 3), alpha]
        lut_lookup_array = np.arange(
            im.shape[0] * im.shape[1]).reshape(im.shape[0], im.shape[1])

        # We can display an color image by using mlab.imshow, a lut color list and
        # a lut lookup table.
        # temporary colormap
        imshow = mlab.imshow(lut_lookup_array,
                             colormap='binary',
                             **kwargs)

        imshow.actor.scale = (scale, scale, 1.0)
        imshow.actor.position = position
        imshow.actor.orientation = orientation
        imshow.module_manager.scalar_lut_manager.lut.table = lut

        return imshow

if __name__ == '__main__':

    from oricreate.api import CreasePatternState, CustomCPFactory, FTV
    from oricreate.crease_pattern import \
        CreasePatternBasesViz3D

    cp = CreasePatternState(X=[[0, 0, 0], [2, 0, 0], [0, 1, 0]],
                            L=[[0, 1], [1, 2], [2, 0]],
                            F=[[0, 1, 2], ])

    cp_factory = CustomCPFactory(formed_object=cp)

    eftlogo_bases = CreasePatternBasesViz3D(
        label='EFT bases', vis3d=cp)

    efttitle_viz3d = FacetsWithImageViz3D(label='EFT title',
                                          F_ref=[0],
                                          N_ref=[0],
                                          F_covered=[[0, 0]],
                                          atimes=[0.0],
                                          im_files=['eft_01.png'],
                                          im_widths=[2],
                                          im_offsets=[[0, 0, 0.001]],
                                          vis3d=cp)

    print('x_refs', efttitle_viz3d.x_refs)
    print('im_files', efttitle_viz3d.im_files)
    print('im_shapes', efttitle_viz3d.im_shapes)
    print('im_scales', efttitle_viz3d.im_scales)
    print('im_centers', efttitle_viz3d.im_centers)
    print('bases', efttitle_viz3d.bases)
    print('rotated bases', efttitle_viz3d.rotated_bases)

    print('glb_centers', efttitle_viz3d.glb_centers)
    print('glb_angles', efttitle_viz3d.glb_euler_angles)

    ftv = FTV()
    ftv.add(cp.viz3d)
    ftv.add(efttitle_viz3d)
    ftv.add(eftlogo_bases)

    m = ftv.mlab
    fig = m.gcf()
    m.figure(figure=fig, bgcolor=(1, 1, 1))
    ftv.plot()

    cp.u[1, 2] = 1.0
    cp.u = np.copy(cp.u)

    cp.viz3d_notify_change()
    efttitle_viz3d.vis3d_changed = True

    print('glb_centers', efttitle_viz3d.glb_centers)
    print('glb_angles', efttitle_viz3d.glb_euler_angles)

    ftv.update(force=True)
    m.show()
