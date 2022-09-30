import unittest
import sympy
import scipy
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

def plot_taylor_sin():
    x = np.linspace(-1, 1)
    for degree in np.array([0, 1, 3, 5, 7]):
        sin_taylor = scipy.interpolate.approximate_taylor_polynomial(np.sin, 0, degree, 1)
        plt.plot(x, sin_taylor(np.pi * x))
    plt.plot(x, np.sin(np.pi * x), color = "black")
    plt.show()

def plot_taylor_e():
    x = np.linspace(-1, 1)
    for degree in np.array([1, 2, 3, 4]):
        exp_taylor = scipy.interpolate.approximate_taylor_polynomial(np.exp, 0, degree, 1)
        plt.plot(x, exp_taylor(x))
    plt.plot(x, np.exp(x), color = "black")
    plt.show()

def plot_taylor_erfc():
    x = np.linspace(-2, 2)
    for degree in [1, 3, 5, 7]:
        erfc_taylor = scipy.interpolate.approximate_taylor_polynomial(scipy.special.erfc, 0, degree, 1)
        plt.plot(x, erfc_taylor(x))
    plt.plot(x, scipy.special.erfc(x), color = "black")
    plt.show()

def plot_monomial_basis():
    x = np.linspace(0, 1)
    for degree in range(0, 11):
        plt.plot(x, x**degree)
    plt.show()

x = sympy.symbols('x')
fun = sympy.sin(sympy.pi * x) #Create symbolic function
numPoints = 1000
px = np.linspace(-1, 1, numPoints) #Splits interval from -1 to 1 into N equally spaced points--to be used for plotting x-values
py = np.zeros(numPoints) #Creates a vector of size N filled with zeros--to be used for plotting y-values
for degree in np.array([0, 1, 3, 5, 7]): #Evaluate taylor expansion to create plot
    t = taylorExpansion(fun, 0, degree) #y
    for i in range(0, numPoints):
        py[i] = t.subs(x, px[i])
    plt.plot(px, py, label = "T(sin(x), degree " + str(degree) + ")") #Create the plot
for i in range(0, numPoints):
    py[i] = fun.subs(x, px[i]) #Evaluate function to create plot
plt.plot(px, py, color = "black", label = "sin(x)")
plt.legend(loc = "upper left")
plt.show()
