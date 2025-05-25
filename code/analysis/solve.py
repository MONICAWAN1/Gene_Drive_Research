# %%
import sympy as sp
from sympy import pprint

q = sp.Symbol('q')
s = sp.Symbol('s')
c = sp.Symbol('c')
h = sp.Symbol('h')

s2 = sp.Symbol('s2')
h2 = sp.Symbol('h2')

# %%
expr_gd = (q**2 * (1-s) + q * (1-q) * (1+c) * (1 - h*s)) / (q**2 * (1-s) + 2 * q * (1-q) * (1 - h*s) + (1-q)**2)
expr_ngd = (q**2 * (1-s2) + q * (1-q) * (1 - h2*s2)) / (q**2 * (1-s2) + 2 * q * (1-q) * (1 - h2*s2) + (1-q)**2)

# %%
sp.simplify(expr_gd - expr_ngd)

q1 = sp.Rational(200, 1000)
q2 = sp.Rational(800, 1000)

eq1 = expr_gd.subs(q, q1) - expr_ngd.subs(q, q1)
eq2 = expr_gd.subs(q, q2) - expr_ngd.subs(q, q2)

sol = sp.solve([eq1, eq2], [s2, h2], dict=True)

s2_sol = sp.simplify(sol[0][s2])
h2_sol = sp.simplify(sol[0][h2])

print("Symbolic expression for s2:")
pprint(s2_sol)

print("\nSymbolic expression for h2:")
pprint(h2_sol)

s2_func = sp.lambdify((s, c, h), s2_sol, modules='numpy')
h2_func = sp.lambdify((s, c, h), h2_sol, modules='numpy')
# %%

def solve_sngd(s, c, h):
    """
    Solve the equation for s2 given s, c, h, and h2.
    
    Parameters:
    s (float): The value of s.
    c (float): The value of c.
    h (float): The value of h.
    h2 (float): The value of h2.
    
    Returns:
    float: The solution for s2.
    """
    return s2_func(s, c, h), h2_func(s, c, h)

print(solve_sngd(0.5, 0.5, 0.3))