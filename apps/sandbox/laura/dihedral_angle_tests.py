'''
Created on Apr 26, 2016

@author: rch
'''

r'''
This is used to show the evaluation steps within the 
deformation energy.
'''
    
from custom_factory_mpl import create_cp_factory


if __name__ == '__main__':
    cp_factory = create_cp_factory()
    cp = cp_factory.formed_object
    print 'x', cp.x

    print 'normed normal vectors of the facets\n', cp.norm_F_normals
    print 'normed normal facet vectors adjacent to the lines\n', cp.norm_iL_F_normals
    
    print 'F_L_vectors_0', cp.F_L_vectors_0
    print 'iL_within_F0', cp.iL_within_F0
    print 'iL_vectors_0', cp.iL_vectors_0
    print 'norm_iL_vectors_0', cp.norm_iL_vectors_0
  
    print 'iL_psi_0', cp.iL_psi_0

    cp.u[5, 2] = 1
    cp.u[4, 2]=  1
    cp.u[7, 2]=  1
    
    cp.u = cp.u
    
    print 'normed normal vectors of the facets\n', cp.norm_F_normals
    print 'normed normal facet vectors adjacent to the lines\n', cp.norm_iL_F_normals
    
    print 'F_L_vectors', cp.F_L_vectors
    print 'iL_within_F0', cp.iL_within_F0
    print 'iL_vectors', cp.iL_vectors
    print 'norm_iL_vectors', cp.norm_iL_vectors
       
    print 'iL_psi2', cp.iL_psi2

