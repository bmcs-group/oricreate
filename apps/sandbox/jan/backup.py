'''
Created on 04.05.2016

@author: jvanderwoerd
'''


        x_1 = ft.x_1
        out = x_1
        fac = cp.F
        
        print out

        nodes = "*Node"
        for i in range(len(out)):   
            temp_node = ' %i \t %.4f \t %.4f \t %.4f\n' % (i + 1, out[i][0], out[i][1], out[i][2])
            temp_node = temp_node.replace('.', ',')
            nodes += temp_node

        facets = "*Elements" 
        for i in range(len(fac)):
            temp_facet = ' %i\tSH36\t%i\t%i\t%i\t\t\t\t\t\t1\n' % (i + 1, fac[i][0] + 1, fac[i][1] + 1, fac[i][2] + 1)
            facets += temp_facet
            
        part = nodes            
        part += facets



        fname = 'spant_dach.inp'
        inp_file = open(fname, 'w')
        inp_file.write(part)
        inp_file.close()
        print'inp file %s written' % fname

