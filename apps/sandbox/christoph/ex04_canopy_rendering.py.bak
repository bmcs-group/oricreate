'''
Created on Feb 21, 2017

@author: rch
'''
from forming_task_anim3d import FTA
from ex04_canopy import DoublyCurvedYoshiFormingProcess, \
    DoublyCurvedYoshiFormingProcessFTV
import numpy as np
from oricreate.api import SimulationHistory

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

    fold_task_clip = False
    load_task_clip = False
    measure_task_clip = True

    if fold_task_clip:
        ft.sim_history.set(anim_t_start=0, anim_t_end=100)
        ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=100)
        ft.sim_history.viz3d['cp'].set(tube_radius=0.005)
        ftv.add(ft.sim_history.viz3d['cp'])
        # ftv.add(ft.sim_history.viz3d['node_numbers'])
        ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
        ftv.add(ft.config.gu['dofs'].viz3d['default'])
        ft.u_1
        
        fta.init_view(a=26.5590138863,
                      e=63.4078712894,
                      d=7.32173763617,
                      f=(1.50299171907,
                         1.16808419892,
                         -0.00931833247854),
                      r=-103.839204821)
        fta.add_cam_move(a=90.104294568,
                         e=63.9434505778,
                         d=8.85930253977,
                         f=(1.54627852193,
                            1.09566457161,
                            -0.0238675464049),
                         r=179.460409553,
                         duration=10, n=30,
                         vot_start=0.0, vot_end=0.0)
        fta.add_cam_move(duration=10, n=30,
                         vot_start=0.0, vot_end=1.0)
        fta.add_cam_move(duration=10, n=30,
                         vot_start=1.0, vot_end=0.0)
        fta.add_cam_move(duration=10, n=30,
                         vot_start=0.0, vot_end=1.0)

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

        fta.init_view(a=92.4328844486,
                      e=86.3390401664,
                      d=6.0,
                      f=(1.45478982231,
                         1.10340822102,
                         -0.0144329282762),
                      r=145.745719523)
        fta.add_cam_move(a=31.6878050709,
                         e=72.0915885759,
                         d=4.15743224624,
                         f=(1.61065357475,
                            1.02715623303,
                            -0.127213985043),
                         r=-99.3637335948,
                         duration=10, n=30,
                         vot_start=0.0, vot_end=0.0)
        fta.add_cam_move(duration=10, n=30,
                         vot_start=0.0, vot_end=1.0)

    if measure_task_clip:

        mt = bsf_process.measure_task

        import os.path as path
        from os.path import expanduser
        home = expanduser("~")

        test_dir = path.join(home, 'simdb', 'exdata',
                             'shell_tests', '2016-09-09-FSH04-Canopy')

        states = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        measured_states = []
        for state in states:
            fname = 'KO%s.txt' % state
            fname = path.join(test_dir, fname)
            print 'read', fname
            measured_state = np.loadtxt(fname)
            x = measured_state[:, 1:]
            measured_states.append(x)

        x_t = np.array(measured_states)
        x_0 = x_t[0, ...]
        u_t = x_t[:, :, :] - x_0[np.newaxis, :, :]

        cp = lt.formed_object
        sh = SimulationHistory(x_0=x_0, L=cp.L, F=cp.F,
                               u_t=u_t)
        sh.set(anim_t_start=0, anim_t_end=50)

        sh.viz3d['displ'].set(tube_radius=0.002)
        ftv.add(sh.viz3d['displ'])

        fta.init_view(a=-5.4936030146,
                          e=86.3293711003,
                          d=3.227514507,
                          f=(1.56419165377,
                             1.25844488005,
                             0.141269125606),
                          r=-91.6518542271)
        clip1 = True
        clip2 = False
        
        if clip1:        
            fta.add_cam_move(a=110.601059374,
                             e=116.278653604,
                             d=2.66736736116,
                             f=(1.54712986896,
                                1.32635250035,
                                0.49588281203),
                             r=36.4265211382,
                             duration=10, n=30,
                             vot_start=0.0, vot_end=0.0)
        elif clip2:
            fta.add_cam_move(a=-52.953825231,
                             e=68.0165837654,
                             d=3.20819908818,
                             f=(2.00831934223,
                                1.45857434599,
                                0.0460539886984),
                             r=-76.4463500663,
                             duration=10, n=30,
                             vot_start=0.0, vot_end=0.0)
        fta.add_cam_move(duration=10, n=30,
                         vot_start=0.0, vot_end=1.0)
        
            
    fta.plot()
    fta.configure_traits()
