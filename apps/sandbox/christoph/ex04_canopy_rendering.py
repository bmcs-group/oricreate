'''
Created on Feb 21, 2017

@author: rch
'''
from forming_task_anim3d import FTA
from ex04_canopy import DoublyCurvedYoshiFormingProcess, \
    DoublyCurvedYoshiFormingProcessFTV
import numpy as np

if __name__ == '__main__':

    bsf_process = DoublyCurvedYoshiFormingProcess(L_x=3.0, L_y=2.41, n_x=4,
                                                  n_y=12, u_x=0.1,
                                                  n_fold_steps=20,
                                                  n_load_steps=20,
                                                  load_factor=5,
                                                  stiffening_bundary=False)

    ftv = DoublyCurvedYoshiFormingProcessFTV(model=bsf_process)

    ft = bsf_process.fold_task
    lt = bsf_process.load_task

    fta = FTA(ftv=ftv)
    fta.init_view(a=33.4389721223,
                  e=61.453898329,
                  d=6,
                  f=(1.58015494765,
                     1.12671403563,
                     -0.111520325399),
                  r=0)

    fold_task_clip = False
    load_task_clip = False
    measure_task_clip = True

    if fold_task_clip:
#         fta.init_view(a=-88.2511452243,
#                       e=66.5983682238,
#                       d=7.47966235448,
#                       f=(1.48817487262,
#                          1.25196069072,
#                          -0.0903056999479),
#                       r=-4.19165336312)
        ft.sim_history.set(anim_t_start=0, anim_t_end=100)
        ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=100)
        ft.sim_history.viz3d['cp'].set(tube_radius=0.005)
        ftv.add(ft.sim_history.viz3d['cp'])
        # ftv.add(ft.sim_history.viz3d['node_numbers'])
        ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
        ftv.add(ft.config.gu['dofs'].viz3d['default'])
        ft.u_1
        fta.add_cam_move(duration=10, n=30)

    if load_task_clip:
        lt.sim_history.set(anim_t_start=0, anim_t_end=100)
        lt.config.gu['dofs'].set(anim_t_start=0, anim_t_end=100)
        lt.config.fu.set(anim_t_start=0, anim_t_end=100)
        lt.sim_history.viz3d['displ'].set(tube_radius=0.005,
                                          warp_scale_factor=5.0)
        # ftv.add(lt.formed_object.viz3d_dict['node_numbers'], order=5)
        ftv.add(lt.sim_history.viz3d['displ'])
        lt.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
        ftv.add(lt.config.gu['dofs'].viz3d['default'])
        ftv.add(lt.config.fu.viz3d['default'])
        lt.config.fu.viz3d['default'].set(anim_t_start=0, anim_t_end=100)
        ftv.add(lt.config.fu.viz3d['node_load'])

        n_max_u = np.argmax(lt.u_1[:, 2])
        cp = lt.formed_object
        iL_phi = cp.iL_psi - cp.iL_psi_0
        iL_m = lt.config._fu.kappa * iL_phi

        fta.add_cam_move(duration=10, n=30)
        
    if measure_task_clip:
        mt = bsf_process.measure_task

        import os.path as path
        from os.path import expanduser
        home = expanduser("~")
        fname = 'KOD.txt'
        test_dir = path.join(home, 'Documents', 'IMB',
                             'KOoricreate')
        fname = path.join(test_dir, fname)
        measured = np.loadtxt(fname)
        node_idx_measured = np.array(measured[:, 0], dtype='int_')
        x_measured = measured[:, 1:]

        cp = mt.formed_object
        x_mes = np.copy(cp.x)
        x_mes[node_idx_measured, :] = x_measured
        u = x_mes - cp.x
        cp.u[:, :] = u

        mt.formed_object.viz3d['displ'].set(tube_radius=0.005)
        ftv.add(mt.formed_object.viz3d['displ'])
        
        fta.add_cam_move(duration=10, n=30)

    fta.plot()
    fta.configure_traits()
