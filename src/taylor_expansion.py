import math
import sympy
import numpy as np
import matplotlib.pyplot as plt
import basis
from scipy import integrate

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

def plot(x, fun, domain, degreeValues, functionLabel):
    numPoints = 1000 #Larger numPoints value refines curves
    px = np.linspace(domain[0], domain[1], numPoints) #Splits interval from left bound to right bound into N equally spaced points--to be used for plotting x-values
    py = np.zeros(numPoints) #Creates a vector of size N filled with zeros--to be used for plotting actual function's y-values
    for degree in degreeValues: #Evaluate taylor expansion to create plot
        t = taylorExpansion(fun, 0, degree) #y-values of taylor expansion
        for i in range(0, numPoints):
            py[i] = t.subs(x, px[i])
        plt.plot(px, py, label = "T(" + functionLabel + ", degree " + str(degree) + ")") #Create the plot
    for i in range(0, numPoints):
        py[i] = fun.subs(x, px[i]) #Evaluate function to create plot
    plt.plot(px, py, color = "black", label = functionLabel)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(loc = "upper left", prop = {"size":6})
    #plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15), ncol = 3, fancybox = True, shadow = True)
    plt.show()

def plotTaylorSin():
    x = sympy.symbols('x')
    fun = sympy.sin(sympy.pi * x) #Create symbolic function
    domain = [-1, 1]
    degreeValues = np.array([0, 1, 3, 5, 7])
    functionLabel = "sin(x)"
    plot(x, fun, domain, degreeValues, functionLabel)

def plotTaylorE():
    x = sympy.symbols('x')
    fun = sympy.exp(x) #Create symbolic function
    domain = [-1, 1]
    degreeValues = np.array([0, 1, 2, 3, 4])
    functionLabel = "e^x"
    plot(x, fun, domain, degreeValues, functionLabel)

def plotTaylorErfc():
    x = sympy.symbols('x')
    fun = sympy.erfc(x) #Create symbolic function
    domain = [-2, 2]
    degreeValues = np.array([0, 1, 3, 5, 7, 9, 11])
    functionLabel = "erfc(x)"
    plot(x, fun, domain, degreeValues, functionLabel)

def plotErrorTaylorErfc():
    x = sympy.symbols('x')
    fun = sympy.erfc(x) #Create symbolic function
    degreeValues = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    error = [] #Initializes empty list to later append to
    for degree in degreeValues:
        t = taylorExpansion(fun, 0, degree)
        domain = [-2, 2]
        error.append(integrate.quad(sympy.lambdify(x, abs(fun - t)), domain[0], domain[1], limit = 100000)[0])
    print(error)
    fig,ax = plt.subplots()
    ax.plot(degreeValues, error)
    plt.yscale("Log")
    #plotError(x, fun, domain, degreeValues, functionLabel)
    plt.show()

#Ask about math.comb

def plotMonomialBasis(highDegree):
    x = np.linspace(0, 1)
    for degree in range(0, highDegree):
        plt.plot(x, x**degree)
    plt.show()

#plotTaylorSin()
#plotTaylorE()
#plotTaylorErfc()
plotErrorTaylorErfc()
