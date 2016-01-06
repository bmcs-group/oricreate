r'''

Fold the twist logo of Engineered Folding Research Center
---------------------------------------------------------

This example shows quadrilateral facets with a twist fold.

@todo: who defines the viz objects.
@todo: provide viz objects also for operators - viz bases, viz normals, 
'''

import os
import string
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from oricreate.api import FTV, FTA
from oricreate.crease_pattern import \
    CreasePatternNormalsViz3D, CreasePatternBasesViz3D
from oricreate.crease_pattern.crease_pattern_viz3d import CreasePatternThickViz3D
from oricreate.gu import GuConstantLength, GuDofConstraints, fix
from oricreate.simulation_step import \
    SimulationStep, SimulationConfig
from sim03_eft_logo_viz3d import \
    FacetsWithTextViz3D, FacetsWithImageViz3D
from sim_task_twist_folding import TwistFolding


def run_sim():

    cp_factory = create_cp_factory()
    cp = cp_factory.formed_object

    if False:
        fig, ax = plt.subplots()
        cp.plot_mpl(ax, facets=True)
        plt.tight_layout()
        plt.show()
        return

    # Configure simulation
    gu_constant_length = GuConstantLength()
    dof_constraints = fix(
        [0], [0, 1, 2]) + fix([1], [1, 2]) + fix([5], [2]) + fix([3], [0], -1.9599)
    gu_dof_constraints = GuDofConstraints(dof_constraints=dof_constraints)
    sim_config = SimulationConfig(gu={'cl': gu_constant_length,
                                      'dofs': gu_dof_constraints})
    sim_step = SimulationStep(forming_task=cp_factory,
                              config=sim_config, acc=1e-5, MAX_ITER=1000)

    if False:
        # Configure rendering visualization operators
        eftlabels_viz3d = FacetsWithTextViz3D(
            label='EFT labels', vis3d=cp)
        efttitle_viz3d = FacetsWithImageViz3D(
            label='EFT title', vis3d=cp,
            F_ref=[0, 2, 4, 6, 8],
            N_ref=[0, 3, 15, 12, 7],
            F_covered=[[0, 9], [2, 11], [4, 13], [6, 15], [8, 17]],
            atimes=[90.0, 180.0, -90, 0, 135],
            im_files=['eft_01.png', 'eft_01.png',
                      'eft_01.png', 'eft_01.png', 'eft_01.png'],
            im_widths=[2, 2, 2, 2, 2],
            im_offsets=[[0, 0, 0.004], [0, 0, 0.004],
                        [0, 0, 0.004], [0, 0, 0.004],
                        [-.295, -.295, 0.004]],
        )
    else:
        fname_eft_front = 'eft_areas_standard_02.png'
        fname_eft_back = 'eft_mission_standard_trans_EF01.png'
        fheight = 0.011
        front_offset = 0.006
        tube_radius = 0.01
        plane_offsets = [0.005, -0.010]
        edge_len = 2.0
        x_offset = 1.0
        c45 = np.cos(np.pi / 4)
        shift_x = x_offset + edge_len / c45 - edge_len
        shift_l = shift_x * c45
        print 'shift_l', shift_l
        F_ref = [0, 2, 4, 6, 8, 16, 1, 12, 14, ]
        N_ref = [0, 3, 15, 12, 7, 4, 1, 11, 13]
        F_covered = [[0, 9], [2, 11], [4, 13], [6, 15],
                     [8, 17], [16, 7], [1, 10], [12, 3], [14, 5]]
        atimes = [90.0, 180.0, -90, 0, 45, 90, 90.0, -90.0, 0.0]
        im_widths = [4, 4, 4, 4, 4, 4, 4, 4, 4]
        imfiles_front = [
            fname_eft_front,
            fname_eft_front,
            fname_eft_front,
            fname_eft_front,
            fname_eft_front,
            fname_eft_front,
            fname_eft_front,
            fname_eft_front,
            fname_eft_front,
        ]
        imfiles_back = [
            fname_eft_back,
            fname_eft_back,
            fname_eft_back,
            fname_eft_back,
            fname_eft_back,
            fname_eft_back,
            fname_eft_back,
            fname_eft_back,
            fname_eft_back,
        ]
        im_front_offsets = [
            [0, 0, front_offset],
            [0, 0, front_offset],
            [0, 0, front_offset],
            [0, 0, front_offset],
            [-shift_l, -shift_l, front_offset],
            [0, -1, front_offset],
            [-2, 0, front_offset],
            [0, -1, front_offset],
            [0, -1, front_offset]
        ]
        im_back_offsets = [
            [0, 0, -fheight],
            [0, 0, -fheight],
            [0, 0, -fheight],
            [0, 0, -fheight],
            [-shift_l, -shift_l, -fheight],
            [0, -1, -fheight],
            [-2, 0, -fheight],
            [0, -1, -fheight],
            [0, -1, -fheight]
        ]
        efttitle_back_viz3d = FacetsWithImageViz3D(
            label='EFT back', vis3d=cp,
            F_ref=F_ref,  # + F_ref,
            N_ref=N_ref,  # + N_ref,
            F_covered=F_covered,  # + F_covered,
            atimes=atimes,  # + atimes,
            im_files=imfiles_back,  # imfiles_front +
            im_widths=im_widths,  # + im_widths,
            im_offsets=im_back_offsets,  # im_front_offsets +
        )
        efttitle_front_viz3d = FacetsWithImageViz3D(
            label='EFT front', vis3d=cp,
            F_ref=F_ref,
            N_ref=N_ref,
            F_covered=F_covered,
            atimes=atimes,
            im_files=imfiles_front,
            im_widths=im_widths,
            im_offsets=im_front_offsets
        )

    eftlogo_normals = CreasePatternNormalsViz3D(
        label='EFT normals', vis3d=cp)
    eftlogo_bases = CreasePatternBasesViz3D(
        label='EFT bases', vis3d=cp)
    eft_thick_viz3d = CreasePatternThickViz3D(
        label='EFT thick', vis3d=cp,
        plane_offsets=plane_offsets)
    cp.viz3d.set(tube_radius=tube_radius)

    # Configure scene
    ftv = FTV()
    ftv.add(cp.viz3d)
    ftv.add(eft_thick_viz3d)
    ftv.add(efttitle_front_viz3d)
    ftv.add(efttitle_back_viz3d)
    # ftv.add(eftlabels_viz3d)
    # ftv.add(eftlogo_normals)
    # ftv.add(eftlogo_bases)

    m = ftv.mlab
    m.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    fig = m.gcf()

    ftv.plot()
    oricreate_mlab_label(m)

    a, e, d, f = m.view()
    e = 135  # 10
    a = 270  # -45  # 120

    f = sim_step.forming_task.formed_object.center
    m.view(a, e, d, f)

    show = False
    if show:
        m.show()

    n_u = 30

    # Control the folding by a fold angle

    phi_range = np.linspace(0.01, np.pi - 0.01, n_u)

    dramaturgy = False
    animation = True

    def circular_folding(phi_range):
        bot_e = 160
        top_e = 10
        if dramaturgy:
            phi_unfolded = phi_range[0] * np.ones_like(phi_range)
            phi_folded = phi_range[-1] * np.ones_like(phi_range)
            phi_range_u = np.hstack(
                [phi_range, phi_folded, phi_range[::-1], phi_unfolded])
        else:
            phi_range_u = phi_range

        u_range = np.cos(phi_range_u) - 1
        phi_range_e = np.linspace(0.01, np.pi - 0.01, n_u * 4)
        azimuth_step = 360.0 / 4 / n_u
        elevation_range = np.cos(
            2 * phi_range_e)**2 * top_e + np.sin(2 * phi_range_e)**2 * bot_e
        time_range = slice(None)
        return u_range, elevation_range, azimuth_step, time_range

    def eft_folding(phi_range):
        n_u12 = 30
        n_c = 60
        n_u23 = 75
        phi_12_range = damped_range(np.pi * 0.3, 0.012, n_u12)
        phi_c = np.ones((n_c,), dtype='float_') * 0.01
        phi_23_range = damped_range(0.01, np.pi - 0.005, n_u23)
        phi_range_u = np.hstack([phi_12_range, phi_c, phi_23_range])
        u_range = np.cos(phi_range_u) - 1
        conf1 = 230, 50, 52, 1.14
        confc = 230, 50, 20, 1.14
        conf2 = 100, 160, 0, 1.1
        conf3 = 270, 179, 0., 0.5
        azimuth_range = np.hstack([damped_range(conf1[0], confc[0], n_u12),
                                   damped_range(confc[0], conf2[0], n_c),
                                   damped_range(conf2[0], conf3[0], n_u23)])
        elevation_range = np.hstack([damped_range(conf1[1], confc[1], n_u12),
                                     damped_range(confc[1], conf2[1], n_c),
                                     damped_range(conf2[1], conf3[1], n_u23)])
        roll_range = np.hstack([damped_range(conf1[2], confc[2], n_u12),
                                damped_range(confc[2], conf2[2], n_c),
                                damped_range(conf2[2], conf3[2], n_u23)])
        d_range = np.hstack([damped_range(conf1[3], confc[3], n_u12) * d,
                             damped_range(confc[3], conf2[3], n_c) * d,
                             damped_range(conf2[3], conf3[3], n_u23) * d])
        time_range = slice(None)
        #time_range = np.array([-1], dtype=int)
        return u_range, elevation_range, azimuth_range, roll_range, time_range, d_range

    def eft_unfolding(phi_range):
        n_u1 = 30
        n_u2 = 60
        n_u3 = 60

        phi_u1 = np.pi - 0.012
        phi_u2 = np.pi * 0.86
        phi_u3 = 0.012

        phi_1 = damped_range(phi_u1, phi_u2, n_u1)
        phi_2 = np.ones((n_u2,), dtype='float_') * phi_u2
        phi_3 = damped_range(phi_u2, phi_u3, n_u3)

        phi_range_u = np.hstack([phi_1, phi_2, phi_3])
        u_range = np.cos(phi_range_u) - 1

        conf1 = 270, 179, 0., 0.5
        conf2 = 230, 140, 60., 0.6
        conf3 = 150, 70, 70., 0.64
        conf4 = 45, 1, 314., 1.14

        azimuth_range = np.hstack([damped_range(conf1[0], conf2[0], n_u1),
                                   damped_range(conf2[0], conf3[0], n_u2),
                                   damped_range(conf3[0], conf4[0], n_u3)])
        elevation_range = np.hstack([damped_range(conf1[1], conf2[1], n_u1),
                                     damped_range(conf2[1], conf3[1], n_u2),
                                     damped_range(conf3[1], conf4[1], n_u3)])
        roll_range = np.hstack([damped_range(conf1[2], conf2[2], n_u1),
                                damped_range(conf2[2], conf3[2], n_u2),
                                damped_range(conf3[2], conf4[2], n_u3)])
        d_range = np.hstack([damped_range(conf1[3], conf2[3], n_u1) * d,
                             damped_range(conf2[3], conf3[3], n_u2) * d,
                             damped_range(conf3[3], conf4[3], n_u3) * d,
                             ])
        time_range = slice(None)
        #time_range = np.array([0], dtype=int)
        return u_range, elevation_range, azimuth_range, roll_range, time_range, d_range

#     u_range, elevation_range, azimuth_range, roll_range, time_range, d_range = eft_folding(
#         phi_range)

    u_range, elevation_range, azimuth_range, roll_range, time_range, d_range = eft_unfolding(
        phi_range)

    tdir = tempfile.mkdtemp()

    fname_list = []
    for i, (u, e, a, r, d) in enumerate(zip(u_range[time_range],
                                            elevation_range[time_range],
                                            azimuth_range[time_range],
                                            roll_range[time_range],
                                            d_range[time_range])):
        gu_dof_constraints.dof_constraints[-1][-1] = u
        sim_step._solve_nr(1.0)
        f = sim_step.forming_task.formed_object.center
        print 'perspective - before', ftv.mlab.view()
        print 'roll', ftv.mlab.roll()
        print 'a,e,d,f', a, e, d, f
        ftv.mlab.view(a, e, d, f)
        ftv.mlab.roll(r)
        print 'perspective', ftv.mlab.view()
        print 'roll', ftv.mlab.roll()
        ftv.update(force=True)
        fname = os.path.join(tdir, 'eftlogo%03d.jpg' % i)
        fname_list.append(fname)
        ftv.mlab.savefig(fname, magnification=2.2)

    if animation:
        animation_file = os.path.join(tdir, 'anim01.gif')
        images = string.join(fname_list, ' ')
        destination = animation_file

        import platform
        if platform.system() == 'Linux':
            os.system(
                'convert -delay 8 -loop 1 ' + images + ' ' + destination)
            # os.system('png2swf -o%s ' % destination + images)
        else:
            raise NotImplementedError(
                'film production available only on linux')
        print 'animation saved in', destination

    ftv.show()

if __name__ == '__main__':
    run_sim()
