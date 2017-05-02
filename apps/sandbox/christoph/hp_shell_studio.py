'''
Created on 23.01.2017
'''
from oricreate.api import FTA, FTV

from apps.examples.ex10_hp_shell import  \
    HPShellFormingProcess, hp_shell_kw_2
import numpy as np


if __name__ == '__main__':

    hp_shell_kw_2 = dict(L_x=10, L_y=10,
                         psi_lines=[
                             10, 23, 35, 40, 7, 20, 41, 44],
                         n_stripes=2,
                         n_steps=10,
                         psi_max=-np.pi / 2.03,
                         fixed_z=[9, 14],
                         fixed_y=[8, 15],
                         fixed_x=[8, 15],
                         link_z=[[8], [15]]
                         )

    bsf_process = HPShellFormingProcess(**hp_shell_kw_2)

    ftv = FTV()

    ft = bsf_process.fold_task

    # ftv.add(it.target_faces[0].viz3d['default'])
    ft.sim_history.set(anim_t_start=0, anim_t_end=100)
    ft.sim_history.viz3d['cp'].set(tube_radius=0.01)
    ftv.add(ft.sim_history.viz3d['cp'])
    #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
#    ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
#    ftv.add(ft.config.gu['dofs'].viz3d['default'])

    ft.u_1

    #ftv.plot()
    #ftv.configure_traits()

    n_cam_move = 20
    fta = FTA(ftv=ftv)
    fta.init_view(a=45.0,
                  e=54.7356103172,
                  d=7.77,
                  f=(0.500000035416,
                     0.287922133088,
                     0.409750220151),
                  r=--120.0)
    fta.add_cam_move(a=24.8592205112,
                     e=35.3839345289,
                     d=4.41536277196,
                     f=(0.500000035416,
                        0.287922133088,
                        0.409750220151),
                     r=-93.3680789265,
                     n=n_cam_move,
                     duration=10,
                     vot_start=0.0, vot_end=1.0,
                     azimuth_move='damped',
                     elevation_move='damped',
                     distance_move='damped')

    fta.plot()
    fta.configure_traits()
