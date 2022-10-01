import unittest
import math
import numpy as np
import sympy

#b1 (C, Image basis)
#b2 (D, Pre-image basis)
#x1 (c)
#x2 (d)
#v1 (Cc)
#v2 (Dd)
#T (Change of basis operator, = D^-1 * C)

def changeOfBasis(b1, b2, x1):
    #Multiply the inverse of b2 by b1 without really taking the inverse of b2
    T = np.linalg.solve(b2, b1) #Row vectors
    #Multiply T by x1
    x2 = np.dot(T, x1) #Column vector
    #x2 = T @ x1
    return x2, T

def evaluateMonomialBasis1D(degree, variate):
    return variate**degree

def evalLegendreBasis1D(degree, variate): #Variate is like the x-value, or in this case, xi
    #set val here
    if degree == 0:
        val = 1.0
    elif degree == 1:
        val = variate
    else:
        i = degree - 1
        term_1 = i * evalLegendreBasis1D(degree = i - 1, variate = variate)
        term_2 = (2 * i + 1) * variate * evalLegendreBasis1D(degree = i, variate = variate)
        val = (term_2 - term_1) / (i + 1)
    return val

def evaluateLagrangeBasis1D(variate, degree, basis_idx):
    nodes = np.linspace(-1, 1, degree + 1) #This divides the domain into degree elements using degree + 1 nodes
    val = 1
    for i in range(0, degree + 1):
        if i != basis_idx:
            numerator = variate - nodes[i]
            denominator = nodes[basis_idx] - nodes[i]
            val *= numerator / denominator
    return val

#Ask about math.comb

def binomialCoefficients(n, k): #n is equal to degree of polynomial, k is the index of the basis function (n + 1 basis functions total)
    numerator = sympy.factorial(n)
    if k == 0:
        denominator = numerator
    else:
        denominator = sympy.factorial(k) * sympy.factorial(n - k)
    return numerator/denominator

def evaluateBernsteinBasis1D(variate, degree, basis_idx): #Defined on interval [0, 1]
    #basis_idx = i (from book equation)
    #degree = p (from book equation)
    coefficient = binomialCoefficients(degree + 1, basis_idx)
    val = coefficient * (variate**basis_idx) * (1 - variate)**(degree - basis_idx)
    return val

class Test_changeOfBasis( unittest.TestCase ):
    def test_standardR2BasisRotate( self ):
        b1 = np.eye(2) #This is a 2x2 identity matrix, row vectors
        b2 = np.array([ [0, 1], [-1, 0] ] ).T #Column vectors; WATCH OUT
        x1 = np.array( [0.5, 0.5] ).T #Column vector

        #b1 = numpy.array([ [1, 4], [3, 1] ]) #Row vectors
        #b2 = numpy.array([ [17, 7], [7, 10] ]) #Row vectors
        #x1 = numpy.array( [2, 5] ).T #Column vector

        #b1 = numpy.array([ [1, 3], [4, 1] ]) #Row vectors
        #b2 = numpy.array([ [1, 0], [0, 1] ]) #Row vectors
        #x1 = numpy.array( [2, 5] ).T #Column vector

        x2, T = changeOfBasis( b1, b2, x1 )
        v1 = b1 @ x1 #@ is for vector-vector multiplication in numpy
        v2 = b2 @ x2
        #R = numpy.linalg.inv(b2) @ b1
        #print( "T = ", T)
        #print("x1 = ", x1)
        #print("x2 = ", x2)
        self.assertTrue( np.allclose( v1, v2 ) ) #allclose is a tolerance
        #self.assertTrue( numpy.allclose( T, R ) )

    def test_standardR2BasisSkew( self ):
        b1 = np.eye(2)
        b2 = np.array([ [0, 1], [0.5, 0.5] ] ).T
        x1 = np.array( [0.5, 0.5] ).T
        x2, T = changeOfBasis( b1, b2, x1 )
        v1 = b1 @ x1
        v2 = b2 @ x2
        self.assertTrue( np.allclose( x2, np.array( [0.0, 1.0] ) ) )
        self.assertTrue( np.allclose( v1, v2 ) )

class Test_evaluateMonomialBasis1D( unittest.TestCase ):
   def test_basisAtBounds( self ):
       self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = 0, variate = 0 ), second = 1.0, delta = 1e-12 )
       for p in range( 1, 11 ):
           self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = p, variate = 0 ), second = 0.0, delta = 1e-12 )
           self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = p, variate = 1 ), second = 1.0, delta = 1e-12 )

   def test_basisAtMidpoint( self ):
       for p in range( 0, 11 ):
           self.assertAlmostEqual( first = evaluateMonomialBasis1D( degree = p, variate = 0.5 ), second = 1 / ( 2**p ), delta = 1e-12 ) 

class Test_evalLegendreBasis1D( unittest.TestCase ):
    def test_basisAtBounds( self ):
        for p in range( 0, 2 ):
            if ( p % 2 == 0 ):
                self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = -1 ), second = +1.0, delta = 1e-12 )
            else:
                self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = -1 ), second = -1.0, delta = 1e-12 )
            self.assertAlmostEqual( first = evalLegendreBasis1D( degree = p, variate = +1 ), second = 1.0, delta = 1e-12 )

    def test_constant( self ):
        for x in np.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 0, variate = x ), second = 1.0, delta = 1e-12 )

    def test_linear( self ):
        for x in np.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 1, variate = x ), second = x, delta = 1e-12 )

    def test_quadratic_at_roots( self ):
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 2, variate = -1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 2, variate = +1.0 / math.sqrt(3.0) ), second = 0.0, delta = 1e-12 )

    def test_cubic_at_roots( self ):
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = -math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D( degree = 3, variate = +math.sqrt( 3 / 5 ) ), second = 0.0, delta = 1e-12 )

class Test_evaluateLagrangeBasis1D( unittest.TestCase ):
    def test_linearLagrange( self ):
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 1, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 1, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 1, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 1, basis_idx = 1 ), second = 1.0, delta = 1e-12 )

    def test_quadraticLagrange( self ):
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = -1, degree = 2, basis_idx = 2 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 1 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate =  0, degree = 2, basis_idx = 2 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateLagrangeBasis1D( variate = +1, degree = 2, basis_idx = 2 ), second = 1.0, delta = 1e-12 )   

class Test_evaluateBernsteinBasis1D( unittest.TestCase ):
    def test_linearBernstein( self ):
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 1, basis_idx = 0 ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 1, basis_idx = 1 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 1, basis_idx = 0 ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 1, basis_idx = 1 ), second = 1.0, delta = 1e-12 )

    def test_quadraticBernstein( self ):
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 0 ), second = 1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 1 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = -1, degree = 2, basis_idx = 2 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 0 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 1 ), second = 0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate =  0, degree = 2, basis_idx = 2 ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 0 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 1 ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evaluateBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 2 ), second = 1.00, delta = 1e-12 )
          
#unittest.main()