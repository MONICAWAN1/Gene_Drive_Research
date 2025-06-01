
#### Version 3: Taylor expansion around q = 0 ######################
#%%
import sympy
from sympy import pprint

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
# simplify the expression and solve it for s2 and h2
# eq is the s2 = ... expression in terms of s, c, h. and q
sympy.simplify(expr_gd - expr_ngd)
eq = sympy.solvers.solve(expr_gd - expr_ngd, s2)[0]
pprint(eq)

#%%
# taylor expansion of equation around q = 0
eq_taylor = sympy.series(eq, q, 0, 3)
print("Taylor expansion of the equation around q = 0:")
pprint(eq_taylor)

# a0 is the constant term, a1 is the coefficient of q
# sngd - a0 - a1 * q = 0
a0 = eq_taylor.coeff(q, 0)
a1 = eq_taylor.coeff(q, 1)

# substitute sngd and hngd into a2 to check the value of the second order term

# solve for s2 and h2 by solving the system of equations: 
# 1) sngd - a0 = 0
# 2) a1 = 0
s2_solved, h2_solved = sympy.solvers.solve([s2-a0, a1], [s2,h2])[0]

print("Symbolic expression for s2:")
pprint(s2_solved)

print("\nSymbolic expression for h2:")
pprint(h2_solved)

s2_func = sympy.lambdify((s, c, h), s2_solved, modules='numpy')
h2_func = sympy.lambdify((s, c, h), h2_solved, modules='numpy')

# print(s2_solved.subs(s, 0.2).subs(c,0.95).subs(h,0.3))
# print(h2_solved.subs(s, 0.2).subs(c,0.95).subs(h,0.3))

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

# %%
### Version 2: Substitute q with specific values and solve for s2 and h2 ##########

# q1 = sp.Rational(200, 1000)
# q2 = sp.Rational(800, 1000)

# eq1 = expr_gd.subs(q, q1) - expr_ngd.subs(q, q1)
# eq2 = expr_gd.subs(q, q2) - expr_ngd.subs(q, q2)

# sol = sp.solve([eq1, eq2], [s2, h2], dict=True)

# s2_sol = sp.simplify(sol[0][s2])
# h2_sol = sp.simplify(sol[0][h2])

# # print("Symbolic expression for s2:")
# # pprint(s2_sol)

# # print("\nSymbolic expression for h2:")
# # pprint(h2_sol)


# s2_func = sp.lambdify((s, c, h), s2_sol, modules='numpy')
# h2_func = sp.lambdify((s, c, h), h2_sol, modules='numpy')
# # %%

# # def solve_sngd(s, c, h):
# #     """
# #     Solve the equation for s2 given s, c, h, and h2.
    
# #     Parameters:
# #     s (float): The value of s.
# #     c (float): The value of c.
# #     h (float): The value of h.
# #     h2 (float): The value of h2.
    
# #     Returns:
# #     float: The solution for s2.
# #     """
# #     return s2_func(s, c, h), h2_func(s, c, h)

# # print(solve_sngd(0.5, 0.5, 0.3))