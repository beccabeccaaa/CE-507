import unittest
import numpy

#Could call image, preimage, domain, etc.
def changeOfBasis( b1, b2, x1 ):
    #Write code to pass test
    x2 = numpy.array( [0.5, 0.5] ).T
    T = 0
    return x2, T #T is our change of basis operator

class Test_changeOfBasis( unittest.TestCase ):
    def test_standardR2BasisRotate( self ):
        b1 = numpy.eye(2)
        b2 = numpy.array([ [0, 1], [-1, 0] ] ).T
        x1 = numpy.array( [0.5, 0.5] ).T
        x2, T = changeOfBasis( b1, b2, x1 )
        v1 = b1 @ x1 #@ is for vector-vector multiplication in numpy
        v2 = b2 @ x2
        self.assertTrue( numpy.allclose( v1, v2 ) ) #allclose is a tolerance

    def test_standardR2BasisSkew( self ):
        b1 = numpy.eye(2)
        b2 = numpy.array([ [0, 1], [0.5, 0.5] ] ).T
        x1 = numpy.array( [0.5, 0.5] ).T
        x2, T = changeOfBasis( b1, b2, x1 )
        v1 = b1 @ x1
        v2 = b2 @ x2
        self.assertTrue( numpy.allclose( x2, numpy.array( [0.0, 1.0] ) ) )
        self.assertTrue( numpy.allclose( v1, v2 ) )