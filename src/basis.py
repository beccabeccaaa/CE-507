import unittest
import numpy

#b1 (C, Image basis)
#b2 (D, Pre-image basis)
#x1 (c)
#x2 (d)
#v1 (Cc)
#v2 (Dd)
#T (Change of basis operator, = D^-1 * C)

def changeOfBasis( b1, b2, x1 ):
    #Multiply the inverse of b2 by b1 without really taking the inverse of b2
    T = numpy.linalg.solve( b2, b1 ) #Row vectors
    #Multiply T by x1
    x2 = numpy.dot( T, x1 ) #Column vector
    #x2 = T @ x1
    return x2, T

class Test_changeOfBasis( unittest.TestCase ):
    def test_standardR2BasisRotate( self ):
        b1 = numpy.eye(2) #This is a 2x2 identity matrix, row vectors
        b2 = numpy.array([ [0, 1], [-1, 0] ] ).T #Column vectors; WATCH OUT
        x1 = numpy.array( [0.5, 0.5] ).T #Column vector

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
        self.assertTrue( numpy.allclose( v1, v2 ) ) #allclose is a tolerance
        #self.assertTrue( numpy.allclose( T, R ) )

    def test_standardR2BasisSkew( self ):
        b1 = numpy.eye(2)
        b2 = numpy.array([ [0, 1], [0.5, 0.5] ] ).T
        x1 = numpy.array( [0.5, 0.5] ).T
        x2, T = changeOfBasis( b1, b2, x1 )
        v1 = b1 @ x1
        v2 = b2 @ x2
        self.assertTrue( numpy.allclose( x2, numpy.array( [0.0, 1.0] ) ) )
        self.assertTrue( numpy.allclose( v1, v2 ) )