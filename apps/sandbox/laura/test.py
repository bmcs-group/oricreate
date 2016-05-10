import numpy as np

#define quantity of facets 
#number of facets=2n-1
n = 4

#create coordinates
        
#number of coordinates
num_coord=2*n+1
                  
#create array
X=np.zeros((num_coord,3))

X[n+1:2*n+1,1]=1

i=1

while i <= n:
    X[i,0]=i
    X[n+i,0]=i
    i = i+1
    
print 'X', X

#create lines

#number of lines
num_lines=4*n-1

#create array
L=np.zeros((num_lines,2))

i=0

while i <= n-1:
    j=3*i
    L[j,0]=i
    L[j,1]=i+1
    i = i+1

i=0

while i+1 <= n:
    j=3*i+1
    L[j,0]=i+1
    L[j,1]=i+n+1
    i = i+1
    
i=0

while i+2 <= n+1:
    j=3*i+2
    L[j,0]=i+1+n
    L[j,1]=i
    i = i+1
    
i=0

while i <= n-2:
    j=i+n+1
    L[3*n+i,0]=j
    k=i+n+2
    L[3*n+i,1]=k
    i = i+1
    
print 'L', L

#create facets
        
#number of facets
num_facet=2*n-1
                  
#create array
F=np.zeros( (num_facet,3) )

i=0

while i <= n-1:
    F[i,0]=i
    F[i,1]=i+1
    F[i,2]=i+n+1
    i = i+1

i=1

while i <= n-1:
    F[n+i-1,0]=i
    F[n+i-1,1]=i+n+1
    F[n+i-1,2]=i+n
    i = i+1
    
print 'F', F
