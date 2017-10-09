'''
Created on Oct 7, 2017

@author: rch
'''


# coding: utf-8

# In[59]:

import numpy as np
import sympy as sp
sp.init_printing()


# In[60]:

a, phi, c, h = sp.symbols('a,phi,c,h')
x1, x2, x3 = sp.symbols('x1,x2,x3')
phi_ = sp.symbols('phi_')
ac_subs = dict(a=3, c=1.0, h=1, phi=0.0001 * sp.pi / 2.0001)


# In[72]:

n1_x = sp.Matrix([sp.cos(phi), 0, sp.sin(phi)])
x = sp.Matrix([x1, x2, x3])
p1 = n1_x.T * (x - a / 2 * n1_x)
p1


# In[73]:

x1_subs = sp.solve(p1, x1)
x1_subs


# In[74]:

n2 = sp.Matrix([1.0, 0, 0])
r2_0 = sp.Matrix([a / 2.0, 0, 0])
p3 = n2.T * (x - r2_0)[0]


# In[75]:

x3_subs = sp.solve(p3.subs(x1_subs), x3)
x3_subs


# In[76]:

p2 = x1 * x1 + x2 * x2 + x3 * x3 - ((a / 2)**2 + h**2)


# In[77]:

x2_ = sp.solve(p2.subs(x1_subs).subs(x3_subs), x2)[1]
x2_


# In[78]:

r3 = sp.simplify(sp.Matrix([x1_subs[x1].subs(x3_subs), x2_, x3_subs[x3]]))
r3.subs(ac_subs).evalf()


# In[82]:

r3_x_r1 = n1_x.cross(r3)
r3_x_r1.subs(ac_subs).evalf()


# In[84]:

f_n = r3_x_r1.normalized()
f_n.subs(ac_subs).evalf()


# In[85]:

print f_n

fn_f_n = sp.lambdify((phi_,), f_n.subs(ac_subs).subs({phi: phi_}).evalf())

print fn_f_n(0.0)
# print 'x', [fn_f_n(xi * sp.pi / 2.0) for xi in [0.0, 0.5, 1.0]]
# xi_range = np.linspace(0, 1, 10)
# print [fn_f_n(xi * sp.pi / 2.0) for xi in xi_range]
