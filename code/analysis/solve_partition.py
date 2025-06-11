#%%
import sympy
from sympy import pprint

#%%
q = sympy.Symbol('q')
s = sympy.Symbol('s')
c = sympy.Symbol('c')
h = sympy.Symbol('h')

s2 = sympy.Symbol('s2')
h2 = sympy.Symbol('h2')

#%%
expr_gd = (q**2 * (1-s) + q * (1-q) * (1+c) * (1 - h*s)) / (q**2 * (1-s) + 2 * q * (1-q) * (1 - h*s) + (1-q)**2)
expr_ngd = (q**2 * (1-s2) + q * (1-q) * (1 - h2*s2)) / (q**2 * (1-s2) + 2 * q * (1-q) * (1 - h2*s2) + (1-q)**2)

sn = 0.5*(1-c)*(1-h*s)
sc = c*(1-h*s)
eq = (2*sn+2*sc-1)/(4*sn+2*sc+s-2)

#%%
# analytical boundary bewteen stable and unstable
dq_0 = sympy.simplify(sympy.diff(expr_gd, q)).subs(q, 0)
dq_1 = sympy.simplify(sympy.diff(expr_gd, q)).subs(q, 1)

dq_0_func = sympy.lambdify((s, c, h), dq_0, modules='numpy')
dq_1_func = sympy.lambdify((s, c, h), dq_1, modules='numpy')

def get_dq(s, c, h):
    return dq_0_func(s, c, h), dq_1_func(s, c, h)
