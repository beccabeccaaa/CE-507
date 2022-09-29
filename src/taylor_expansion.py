#from gettext import npgettext
import sympy
import numpy as np
import matplotlib.pyplot as plt

#a is the value at which the expansion is centered
#x is the variable that remains
#t is the taylor expansion

def taylorExpansion(fun, a, order):
    x = list(fun.atoms(sympy.Symbol))[0]
    t = 0
    for i in range(0, order + 1):
        df = sympy.diff(fun, x, i) #Take derivative with variables (take the ith derivative of fun with respect to x) (Next line: and evaluate at a)
        term = (df.subs(x, a) / sympy.factorial(i) ) * (x - a)**i
        t += term
        print(t)
    return t

#Create symbolic function & taylor expansion
x = sympy.symbols('x')
fun = sympy.sin(sympy.pi * x)
t = taylorExpansion(fun, 0, 4) #y

#Evaluate function and taylor expansion to create plot data
N = 1000
px = np.linspace(-1, 1, N)
py = np.zeros(N)
fy = np.zeros(N)
for i in range(0, N ):
    fy[i] = fun.subs(x, px[i])
    py[i] = t.subs(x, px[i])

#Create the plot
fig, ax = plt.subplots()
ax.plot(px, fy, linewidth=2.0)
ax.plot(px, py, linewidth=2.0)
plt.show()
