import unittest
import numpy

#Could call image, preimage, domain, etc.
def changeOfBasis( b1, b2, x1 ): #b1 is the preimage basis, b2 is the image basis
    #Write code to pass test
    T = numpy.linalg.solve( b2, b1 ) #Multiply the inverse of b1 by b2
    x2 = numpy.dot( T, x1 ) #Multiply T by x1
    return x2, T #T is our change of basis operator

class Test_changeOfBasis( unittest.TestCase ):
    def test_standardR2BasisRotate( self ):
        b1 = numpy.eye(2) #This is a 2x2 identity matrix
        b2 = numpy.array([ [0, 1], [-1, 0] ] ).T
        x1 = numpy.array( [0.5, 0.5] ).T
        x2, T = changeOfBasis( b1, b2, x1 )
        v1 = b1 @ x1 #@ is for vector-vector multiplication in numpy
        v2 = b2 @ x2
        #By hand, I should determine what T should be, add a test for it, add correct assert (mayybe self.assertTrue(nump.allclose(hard coded T)))
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