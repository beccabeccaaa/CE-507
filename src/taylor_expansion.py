#from gettext import npgettext
import sympy
import matplotlib.pyplot as plt
import numpy as np

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

x = np.linspace(-5 ,5 ,100)
y = taylorExpansion(np.sin(np.pi * x), 0, 2)

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# plot the function
plt.plot(x, y, 'r')

# show the plot
plt.show()