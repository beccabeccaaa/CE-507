import sympy

#a is the value at which the expansion is centered
#x is the variable that remains
#t is the taylor expansion

def taylorExpansion( fun, a, order ):
    x = list( fun.atoms( sympy.Symbol ) )[0]
    t = 0
    for i in range( 0, order + 1 ):
        dsf = sympy.diff( fun, x, i ) #Take derivative with variables (take the ith derivative of fun with respect to x) (Next line: and evaluate at a)
        term = ( df.subs( x, a ) / sympy.factorial( i ) ) * ( x - a )**i
        t += term
    return t