'''
Created on Oct 7, 2017

@author: rch
'''
from sympy.solvers.solveset import nonlinsolve
import sympy as sp

sp.init_printing()

a, b, h = sp.symbols('a, b, h')
v1, u2, w1, w2 = sp.symbols('v1, u2, w1, w2')

R1 = sp.Matrix([a, h, 0]) + sp.Matrix([0, v1, w1])
R2 = sp.Matrix([b, 0, 0]) + sp.Matrix([u2, 0, w2])
R3 = sp.Matrix([0, 0, 0])

V12 = R2 - R1
V23 = R3 - R2
V31 = R1 - R3

L12 = h**2 + (b - a)**2
L23 = b**2
L31 = h**2 + a**2

EQ1 = (V12.T * V12)[0, 0] - L12
EQ2 = (V23.T * V23)[0, 0] - L23
EQ3 = (V31.T * V31)[0, 0] - L31

EQS = [EQ1, EQ2, EQ3]
var = [v1, u2, w2]
print EQ1
print EQ2
print EQ3
print

sol = nonlinsolve(EQS, var)

for s in sol:
    print s
