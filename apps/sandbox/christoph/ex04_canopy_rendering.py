'''
Created on Feb 21, 2017

@author: rch
'''


from forming_task_anim3d import FTA
from oricreate.export import \
    InfoCadMeshExporter, ScaffoldingExporter
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

    fa = bsf_process.factory_task
    mt = bsf_process.mask_task
    ab = bsf_process.add_boundary_task

    if False:
        import pylab as p
        ax = p.axes()
        ab.formed_object.plot_mpl(ax)
        p.show()

    it = bsf_process.init_displ_task
    ft = bsf_process.fold_task
    tt = bsf_process.turn_task
    tt2 = bsf_process.turn_task2
    lt = bsf_process.load_task      

    fta = FTA(ftv=ftv)

    if False: # This was the old main
        
        animate = False
        show_init_task = False
        show_fold_task = True
        show_turn_task = False
        show_turn_task2 = False
        show_load_task = False
        show_measure_task = False
        export_and_show_mesh = False
        export_scaffolding = False
    
        fta.init_view(a=33.4389721223,
                      e=61.453898329,
                      d=5.0,
                      f=(1.58015494765,
                         1.12671403563,
                         -0.111520325399),
                      r=-105.783218753)
        
        if show_init_task:
            ftv.add(it.target_faces[0].viz3d['default'])
            it.formed_object.viz3d['cp'].set(tube_radius=0.002)
            ftv.add(it.formed_object.viz3d['cp'])
            #ftv.add(it.formed_object.viz3d['node_numbers'], order=5)
            it.u_1
    
        if show_fold_task:
            ft.sim_history.set(anim_t_start=0, anim_t_end=10)
            ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=5)
            ft.sim_history.viz3d['cp'].set(tube_radius=0.002)
            ftv.add(ft.sim_history.viz3d['cp'])
    #        ftv.add(ft.sim_history.viz3d['node_numbers'])
            ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
            ftv.add(ft.config.gu['dofs'].viz3d['default'])
            ft.u_1
    
            fta.add_cam_move(duration=10, n=20)
    
        if show_turn_task:
            tt.formed_object.set(anim_t_start=10, anim_t_end=20)
            tt.formed_object.viz3d['cp'].set(tube_radius=0.002)
            ftv.add(tt.formed_object.viz3d['cp'])
    
            fta.add_cam_move(duration=10, n=20,
                             )
    
        if show_turn_task2:
            tt2.u_1
            tt2.formed_object.set(anim_t_start=10, anim_t_end=20)
            tt2.sim_history.viz3d['cp'].set(tube_radius=0.002)
            ftv.add(tt2.sim_history.viz3d['cp'])
            tt2.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
            ftv.add(tt2.config.gu['dofs'].viz3d['default'])
            fta.add_cam_move(a=45, e=73, d=5,
                             duration=10, n=20,
                             azimuth_move='damped',
                             elevation_move='damped',
                             distance_move='damped')
    
        if show_load_task == True:
            lt.sim_history.set(anim_t_start=20, anim_t_end=50)
            lt.config.gu['dofs'].set(anim_t_start=20, anim_t_end=50)
            lt.config.fu.set(anim_t_start=20, anim_t_end=50)
            lt.sim_history.viz3d['displ'].set(tube_radius=0.002,
                                              warp_scale_factor=5.0)
            #    ftv.add(lt.formed_object.viz3d_dict['node_numbers'], order=5)
            ftv.add(lt.sim_history.viz3d['displ'])
            lt.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
            ftv.add(lt.config.gu['dofs'].viz3d['default'])
            ftv.add(lt.config.fu.viz3d['default'])
            lt.config.fu.viz3d['default'].set(anim_t_start=30, anim_t_end=50)
            ftv.add(lt.config.fu.viz3d['node_load'])
    
            print 'u_13', lt.u_1[13, 2]
            n_max_u = np.argmax(lt.u_1[:, 2])
            print 'node max_u', n_max_u
            print 'u_max', lt.u_1[n_max_u, 2]
    
            cp = lt.formed_object
            iL_phi = cp.iL_psi - cp.iL_psi_0
            iL_m = lt.config._fu.kappa * iL_phi
            print 'moments', np.max(np.fabs(iL_m))
    
            fta.add_cam_move(duration=10, n=20)
            fta.add_cam_move(duration=10, n=20, vot_start=1.0)
            fta.add_cam_move(duration=10, n=20, vot_start=1.0)
    
        if show_measure_task:
            mt = bsf_process.measure_task
    
            import os.path as path
            from os.path import expanduser
            home = expanduser("~")
            fname = 'KO8.txt'
            test_dir = path.join(home, 'simdb', 'exdata',
                                 'shell_tests', '2016-09-09-FSH04-Canopy')
            fname = path.join(test_dir, fname)
            measured = np.loadtxt(fname)
            node_idx_measured = np.array(measured[:, 0], dtype='int_')
            x_measured = measured[:, 1:]
    
            cp = mt.formed_object
            x_mes = np.copy(cp.x)
            x_mes[node_idx_measured, :] = x_measured
            u = x_mes - cp.x
            cp.u[:, :] = u
    
            mt.formed_object.viz3d['displ'].set(tube_radius=0.002)
            ftv.add(mt.formed_object.viz3d['displ'])
    
        if export_and_show_mesh:
            lt = bsf_process.load_task
            me = InfoCadMeshExporter(forming_task=lt, n_l_e=4)
            me.write()
            X, F = me._get_geometry()
            x, y, z = X.T
            import mayavi.mlab as m
            me.plot_mlab(m)
            m.show()
    
        if export_scaffolding:
            sf = ScaffoldingExporter(forming_task=ft)
            
        if animate:
            n_cam_move = 20
            fta = FTA(ftv=ftv)
            fta.init_view(a=33.4389721223,
                          e=61.453898329,
                          d=4.13223140496, f=(1.58015494765,
                                              1.12671403563,
                                              -0.111520325399), r=-105.783218753)
                
            fta.add_cam_move(a=60, e=70, n=n_cam_move, d=8, r=-120,
                             duration=10,
                             vot_fn=lambda cmt: np.linspace(0.01, 0.5, n_cam_move),
                             azimuth_move='damped',
                             elevation_move='damped',
                             distance_move='damped')
            fta.add_cam_move(a=80, e=80, d=7, n=n_cam_move, r=-132,
                             duration=10,
                             vot_fn=lambda cmt: np.linspace(0.5, 1.0, n_cam_move),
                             azimuth_move='damped',
                             elevation_move='damped',
                             distance_move='damped')
    
            fta.plot()
            fta.configure_traits()
    
    '''scripted scenes:'''
    scene1 = True
            
    if scene1:
        fta.init_view(a=-88.2511452243,
                      e=66.5983682238,
                      d=7.47966235448,
                      f=(1.48817487262,
                         1.25196069072,
                         -0.0903056999479),
                      r=-4.19165336312)
        ft.sim_history.set(anim_t_start=0, anim_t_end=100)
        ft.config.gu['dofs'].set(anim_t_start=0, anim_t_end=100)
        ft.sim_history.viz3d['cp'].set(tube_radius=0.005)
        ftv.add(ft.sim_history.viz3d['cp'])
#        ftv.add(ft.sim_history.viz3d['node_numbers'])
        ft.config.gu['dofs'].viz3d['default'].scale_factor = 0.5
        ftv.add(ft.config.gu['dofs'].viz3d['default'])
        ft.u_1
        fta.add_cam_move(duration=10, n=30)

    fta.plot()
    fta.configure_traits()
