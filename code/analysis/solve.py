# %%
import sympy

q = sympy.Symbol('q')
s = sympy.Symbol('s')
c = sympy.Symbol('c')
h = sympy.Symbol('h')

s2 = sympy.Symbol('s2')
h2 = sympy.Symbol('h2')

# %%
expr_gd = (q**2 * (1-s) + q * (1-q) * (1+c) * (1 - h*s)) / (q**2 * (1-s) + 2 * q * (1-q) * (1 - h*s) + (1-q)**2)
expr_ngd = (q**2 * (1-s2) + q * (1-q) * (1 - h2*s2)) / (q**2 * (1-s2) + 2 * q * (1-q) * (1 - h2*s2) + (1-q)**2)

# %%
sympy.simplify(expr_gd - expr_ngd)
s2_sol = sympy.solvers.solve(expr_gd - expr_ngd, s2)[0]

s2_func = sympy.lambdify((s, c, h, h2, q), s2_sol, modules='numpy')

def solve_sngd(s, c, h, h2, q):
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
    return s2_func(s, c, h, h2, q)