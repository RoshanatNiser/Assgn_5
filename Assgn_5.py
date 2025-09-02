# Assignment 4: Gauss_seidel
# Name: Roshan Yadav
# Roll no: 2311144

from Func_lib_assgn_5 import *

# Question_1: Cholesky decomposition and Gauss-Seidel
print("Solving Question 1")
A=read_matrix('A.txt')
b=read_vector('b.txt')
print("\nOutput of Cholesky decomposition",cholesky_solve(A,b)) #output:[1.0, 0.9999999999999999, 1.0, 1.0, 1.0, 1.0]
print("\nOutput of Jacobi:",jacobi(A,b))                        #output:[0.9999989753998146, 0.9999985509965219, 0.9999989753998146, 0.9999989753998146, 0.9999985509965219, 0.9999989753998146]



#Question_2: Jacobi and Gauss-Seidel.
print("\nSolving Question 2")
C=read_matrix('C.txt')
d=read_vector('d.txt')
print("\nOutput of Jacobi:",jacobi(C,d))             #output:[2.9791649583226008, 2.215599258220273, 0.21128373337161171, 0.15231661140963978, 5.71503326456748]
print("\n Output of gauss-seidel",gauss_seidel(C,d)) #output:[2.979165086347139, 2.215599676186742, 0.21128402698819165, 0.15231700827754793, 5.715033568811629]
