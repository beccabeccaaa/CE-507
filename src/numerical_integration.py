import unittest
import math
import numpy as np
import sympy
import basis

def getRiemannQuadrature(num_points):
    if num_points < 1:
        raise(Exception("num_points must be at least 1"))
    num_bounds = num_points + 1
    x = np.linspace(-1, 1, num_points + num_bounds)
    xQuadrature = x[1::2] #Take every second index starting from index 1
    wQuadrature = np.diff(x[0::2]) #Take every second index starting from index 0
    return xQuadrature, wQuadrature

def computeRiemannQuadrature(fun, num_points):
    xQuadrature, wQuadrature = getRiemannQuadrature(num_points = num_points)
    integral = 0.0
    for i in range( 0, num_points ):
        integral += fun(xQuadrature[i]) * wQuadrature[i]
    return integral

def getGaussLegendreQuadrature(num_points):
    if num_points == 1:
        x = [ 0.0 ]
        w = [ 2.0 ]
    elif num_points == 2:
        x = [ -1.0 / math.sqrt(3), 
              +1.0 / math.sqrt(3) ]

        w = [ 1.0, 
              1.0  ]
    elif num_points == 3:
        x = [ -1.0 * math.sqrt(3.0 / 5.0), 
               0.0, 
              +1.0 * math.sqrt(3.0 / 5.0) ]

        w = [ 5.0 / 9.0, 
              8.0 / 9.0, 
              5.0 / 9.0 ]
    elif num_points == 4:
        x = [ -1.0 * math.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ),
              -1.0 * math.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ),
              +1.0 * math.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ),
              +1.0 * math.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * math.sqrt( 6.0 / 5.0 ) ) ]
        
        w = [ ( 18.0 - math.sqrt( 30.0 ) ) / 36.0,
              ( 18.0 + math.sqrt( 30.0 ) ) / 36.0,
              ( 18.0 + math.sqrt( 30.0 ) ) / 36.0,
              ( 18.0 - math.sqrt( 30.0 ) ) / 36.0 ]
    elif num_points == 5:
        x = [ -1.0 / 3.0 * math.sqrt( 5.0 + 2.0 * math.sqrt( 10.0 / 7.0 ) ),
              -1.0 / 3.0 * math.sqrt( 5.0 - 2.0 * math.sqrt( 10.0 / 7.0 ) ),
               0.0,
              +1.0 / 3.0 * math.sqrt( 5.0 - 2.0 * math.sqrt( 10.0 / 7.0 ) ),
              +1.0 / 3.0 * math.sqrt( 5.0 + 2.0 * math.sqrt( 10.0 / 7.0 ) ) ]
        
        w = [ ( 322.0 - 13.0 * math.sqrt( 70.0 ) ) / 900.0,
              ( 322.0 + 13.0 * math.sqrt( 70.0 ) ) / 900.0,
                128.0 / 225.0,
              ( 322.0 + 13.0 * math.sqrt( 70.0 ) ) / 900.0,
              ( 322.0 - 13.0 * math.sqrt( 70.0 ) ) / 900.0, ]
    else:
        raise( Exception( "num_points_MUST_BE_INTEGER_IN_[1-5]" ) )
    return x, w




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

