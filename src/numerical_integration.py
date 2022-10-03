import unittest
import math
import numpy as np
import sympy
#import scipy
import basis

def getRiemannQuadrature(num_points): #This function returns the abscissae and weights for Riemann quadrature
    if num_points < 1:
        raise(Exception("num_points_MUST_BE_INTEGER_GEQ_1")) #FIXME Why?
    num_boundary_points = num_points + 1 #These points are on each side of each middle point
    allPoints = np.linspace(-1, 1, num_points + num_boundary_points)
    xQuadrature = allPoints[1::2] #Take every second index starting from index 1 (essentially, all the midpoints)
    wQuadrature = np.diff(allPoints[0::2]) #Take every second index starting from index 0, take distance between(essentially, the distance between all the points on the side of each midpoint)
    return xQuadrature, wQuadrature

def riemannQuadrature(fun, num_points):
    xQuadrature, wQuadrature = getRiemannQuadrature(num_points)
    integral = 0.0
    for i in range(0, num_points):
        integral += fun(xQuadrature[i]) * wQuadrature[i]
    return integral

def getNewtonCotesQuadrature(num_points):
    if (num_points < 1) or (num_points > 6):
        raise(Exception("num_points_MUST_BE_INTEGER_IN_[1,6]"))
    if num_points == 1:
        xQuadrature = np.array([0.0])
        wQuadrature = np.array([2.0])
    elif num_points == 2:
        xQuadrature = np.array([-1.0, +1.0])
        wQuadrature = np.array([1.0, 1.0])
    elif num_points == 3:
        xQuadrature = np.array([-1.0, 0.0, +1.0])
        wQuadrature = np.array([1.0, 4.0, 1.0]) / 3.0
    elif num_points == 4:
        xQuadrature = np.array([-1.0, -1.0/3.0, +1.0/3.0, +1.0])
        wQuadrature = np.array([1.0, 3.0, 3.0, 1.0]) / 4.0
    elif num_points == 5:
        xQuadrature = np.array([-1.0, -0.5, 0.0, +0.5, +1.0])
        wQuadrature = np.array([7.0, 32.0, 12.0, 32.0, 7.0]) / 45.0
    elif num_points == 6:
        xQuadrature = np.array([-1.0, -0.6, -0.2, +0.2, +0.6, +1.0])
        wQuadrature = np.array([19.0, 75.0, 50.0, 50.0, 75.0, 19.0]) / 144.0
    return xQuadrature, wQuadrature

def computeNewtonCotesQuadrature(fun, num_points):
    xQuadrature, wQuadrature = getNewtonCotesQuadrature( num_points = num_points )
    integral = 0.0
    for i in range(0, len(xQuadrature)):
        integral += fun(xQuadrature[i]) * wQuadrature[i]
    return integral

def computeGaussLegendreQuadrature(n):
    M = np.zeros(2 * n, dtype = "double")
    M[0] = 2.0
    x0 = np.linspace(-1, 1, n)
    sol = scipy.optimize.least_squares(lambda x : objFun(M, x), x0, bounds = (-1, 1), ftol = 1e-14, xtol = 1e-14, gtol = 1e-14)
    qp = sol.x
    w = solveLinearMomentFit(M, qp)
    return qp, w

def assembleLinearMomentFitSystem(degree, pts):
    A = np.zeros(shape = (degree + 1, len(pts)), dtype = "double")
    for i in range(0, degree + 1):
        for j in range(0, len(pts)):
            A[i, j] = basis.evalLegendreBasis1D(degree = i, variate = pts)
    return A

def solveLinearMomentFit(M, pts): #Solved P * d = f for d
    degree = len(M) - 1
    A = assembleLinearMomentFitSystem(degree, pts)
    sol = scipy.optimize.lsq_linear(A, M)
    w = sol.x
    return w

def objFun(M, pts):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem(degree, pts)
    w = solveLinearMomentFit(M, pts)
    obj_val = np.squeeze(M - A @ w)
    #print(A.shape, w.shape, obj_val.shape)
    return obj_val

unittest.main()

class Test_getRiemannQuadrature( unittest.TestCase ):
    def test_zero_points( self ):
        with self.assertRaises( Exception ) as context:
            getRiemannQuadrature( num_points = 0 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_GEQ_1", str( context.exception ) )

    def test_one_point( self ):
        x, w = getRiemannQuadrature( num_points = 1 )
        self.assertAlmostEqual( first = x, second = 0.0 )
        self.assertAlmostEqual( first = w, second = 2.0 )
        self.assertIsInstance( obj = x, cls = np.ndarray )
        self.assertIsInstance( obj = w, cls = np.ndarray )

    def test_two_point( self ):
        x, w = getRiemannQuadrature( num_points = 2 )
        self.assertTrue( np.allclose( x, [ -0.50, 0.50 ] ) )
        self.assertTrue( np.allclose( w, [ 1.0, 1.0 ] ) )
        self.assertIsInstance( obj = x, cls = np.ndarray )
        self.assertIsInstance( obj = w, cls = np.ndarray )

    def test_three_point( self ):
        x, w = getRiemannQuadrature( num_points = 3 )
        self.assertTrue( np.allclose( x, [ -2.0/3.0, 0.0, 2.0/3.0 ] ) )
        self.assertTrue( np.allclose( w, [ 2.0/3.0, 2.0/3.0, 2.0/3.0 ] ) )
        self.assertIsInstance( obj = x, cls = np.ndarray )
        self.assertIsInstance( obj = w, cls = np.ndarray )

    def test_many_points( self ):
        for num_points in range( 1, 100 ):
            x, w = getRiemannQuadrature( num_points = num_points )
            self.assertTrue( len( x ) == num_points )
            self.assertTrue( len( w ) == num_points )
            self.assertIsInstance( obj = x, cls = np.ndarray )
            self.assertIsInstance( obj = w, cls = np.ndarray )

class Test_computeRiemannQuadrature( unittest.TestCase ):
    def test_integrate_constant_one( self ):
        constant_one = lambda x : 1
        for num_points in range( 1, 100 ):
            self.assertAlmostEqual( first = riemannQuadrature( fun = constant_one, num_points = num_points ), second = 2.0, delta = 1e-12 )

    def test_integrate_linear( self ):
        linear = lambda x : x
        for num_points in range( 1, 100 ):
            self.assertAlmostEqual( first = riemannQuadrature( fun = linear, num_points = num_points ), second = 0.0, delta = 1e-12 )

    def test_integrate_quadratic( self ):
        linear = lambda x : x**2
        error = []
        for num_points in range( 1, 100 ):
            error.append( abs( (2.0 / 3.0) - riemannQuadrature( fun = linear, num_points = num_points ) ) )
        self.assertTrue( np.all( np.diff( error ) <= 0.0 ) )

    def test_integrate_sin( self ):
        sin = lambda x : math.sin(x)
        error = []
        for num_points in range( 1, 100 ):
            self.assertAlmostEqual( first = riemannQuadrature( fun = sin, num_points = num_points ), second = 0.0, delta = 1e-12 )

    def test_integrate_cos( self ):
        cos = lambda x : math.cos(x)
        error = []
        for num_points in range( 1, 100 ):
            error.append( abs( (2.0 / 3.0) - riemannQuadrature( fun = cos, num_points = num_points ) ) )
        self.assertTrue( np.all( np.diff( error ) <= 0.0 ) )

class Test_getNewtonCotesQuadrature( unittest.TestCase ):
    def test_incorrect_num_points( self ):
        with self.assertRaises( Exception ) as context:
            getNewtonCotesQuadrature( num_points = 0 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1,6]", str( context.exception ) )
        with self.assertRaises( Exception ) as context:
            getNewtonCotesQuadrature( num_points = 7 )
        self.assertEqual( "num_points_MUST_BE_INTEGER_IN_[1,6]", str( context.exception ) )

    def test_return_types( self ):
        for num_points in range( 1, 7 ):
            x, w = getNewtonCotesQuadrature( num_points = num_points )
            self.assertIsInstance( obj = x, cls = np.ndarray )
            self.assertIsInstance( obj = w, cls = np.ndarray )
            self.assertTrue( len( x ) == num_points )
            self.assertTrue( len( w ) == num_points )

class Test_computeNewtonCotesQuadrature( unittest.TestCase ):
    def test_integrate_constant_one( self ):
        constant_one = lambda x : 1 * x**0
        for degree in range( 1, 6 ):
            num_points = degree + 1
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = constant_one, num_points = num_points ), second = 2.0, delta = 1e-12 )

    def test_exact_poly_int( self ):
        for degree in range( 1, 6 ):
            num_points = degree + 1
            poly_fun = lambda x : ( x + 1.0 ) ** degree
            indef_int = lambda x : ( ( x + 1 ) ** ( degree + 1) ) / ( degree + 1 )
            def_int = indef_int(1.0) - indef_int(-1.0)
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = poly_fun, num_points = num_points ), second = def_int, delta = 1e-12 )

    def test_integrate_sin( self ):
        sin = lambda x : math.sin(x)
        for num_points in range( 1, 7 ):
            self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = sin, num_points = num_points ), second = 0.0, delta = 1e-12 )

    def test_integrate_cos( self ):
        cos = lambda x : math.cos(x)
        self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = cos, num_points = 6 ), second = 2*math.sin(1), delta = 1e-4 )

class Test_computeGaussLegendreQuadrature( unittest.TestCase ):
    def test_1_pt( self ):
        qp_gold = np.array( [ 0.0 ] )
        w_gold = np.array( [ 2.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 1 )
        self.assertAlmostEqual( first = qp, second = qp_gold, delta = 1e-12 )
        self.assertAlmostEqual( first = w, second = w_gold, delta = 1e-12 )

    def test_2_pt( self ):
        qp_gold = np.array( [ -1.0/np.sqrt(3), 1.0/np.sqrt(3) ] )
        w_gold = np.array( [ 1.0, 1.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 2 )
        self.assertTrue( np.allclose( qp, qp_gold ) )
        self.assertTrue( np.allclose( w, w_gold ) )

    def test_3_pt( self ):
        qp_gold = np.array( [ -1.0 * np.sqrt( 3.0 / 5.0 ),
                                0.0,
                                +1.0 * np.sqrt( 3.0 / 5.0 ) ] )
        w_gold = np.array( [ 5.0 / 9.0,
                                8.0 / 9.0,
                                5.0 / 9.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 3 )
        self.assertTrue( np.allclose( qp, qp_gold ) )
        self.assertTrue( np.allclose( w, w_gold ) )

    def test_4_pt( self ):
        qp_gold = np.array( [ -1.0 * np.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * np.sqrt( 6.0 / 5.0 ) ),
                                -1.0 * np.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * np.sqrt( 6.0 / 5.0 ) ),
                                +1.0 * np.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * np.sqrt( 6.0 / 5.0 ) ),
                                +1.0 * np.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * np.sqrt( 6.0 / 5.0 ) ) ] )
        w_gold = np.array( [ ( 18.0 - np.sqrt( 30.0 ) ) / 36.0,
                                ( 18.0 + np.sqrt( 30.0 ) ) / 36.0,
                                ( 18.0 + np.sqrt( 30.0 ) ) / 36.0,
                                ( 18.0 - np.sqrt( 30.0 ) ) / 36.0 ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 4 )
        self.assertTrue( np.allclose( qp, qp_gold ) )
        self.assertTrue( np.allclose( w, w_gold ) )

    def test_5_pt( self ):
        qp_gold = np.array( [ -1.0 / 3.0 * np.sqrt( 5.0 + 2.0 * np.sqrt( 10.0 / 7.0 ) ),
                                -1.0 / 3.0 * np.sqrt( 5.0 - 2.0 * np.sqrt( 10.0 / 7.0 ) ),
                                0.0,
                                +1.0 / 3.0 * np.sqrt( 5.0 - 2.0 * np.sqrt( 10.0 / 7.0 ) ),
                                +1.0 / 3.0 * np.sqrt( 5.0 + 2.0 * np.sqrt( 10.0 / 7.0 ) ) ] )
        w_gold = np.array( [ ( 322.0 - 13.0 * np.sqrt( 70.0 ) ) / 900.0,
                                ( 322.0 + 13.0 * np.sqrt( 70.0 ) ) / 900.0,
                                128.0 / 225.0,
                                ( 322.0 + 13.0 * np.sqrt( 70.0 ) ) / 900.0,
                                ( 322.0 - 13.0 * np.sqrt( 70.0 ) ) / 900.0, ] )
        [ qp, w ] = computeGaussLegendreQuadrature( 5 )
        self.assertTrue( np.allclose( qp, qp_gold ) )
        self.assertTrue( np.allclose( w, w_gold ) )
