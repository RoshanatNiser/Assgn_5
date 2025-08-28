# Assignment 4: Gauss_seidel
# Name: Roshan Yadav
# Roll no: 2311144

from Func_lib_assgn_5 import *

# Question_1: Cholesky decomposition and Gauss-Seidel
print("Solving Question 1")
A=read_matrix('A.txt')
b=read_vector('b.txt')
print("\nOutput of Cholesky decomposition",cholesky_solve(A,b))
print("\nOutput of Jacobi:",jacobi(A,b))



#Question_2: Jacobi and Gauss-Seidel.
print("\nSolving Question 2")
C=read_matrix('C.txt')
d=read_vector('d.txt')
print("\nOutput of Jacobi:",jacobi(C,d))
print("\n Output of gauss-seidel",gauss_seidel(C,d))

