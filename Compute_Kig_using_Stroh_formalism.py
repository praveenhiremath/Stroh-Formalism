from __future__ import division
import numpy as np
import io
import scipy
from array import *
import os
import math
import scipy.linalg as la
from numpy.linalg import inv
from aniso_elastic_to_compliance import *

'''
The code is for PhD research purposes

Author: Praveenkumar Hiremath
Email: praveenkumar.hiremath@mek.lth.se (Email at the University)
       praveenkumar.hiremath2911@gmail.com (Private email)
'''


S=np.loadtxt('rotated_compliance.dat')
prop_array=list(map(int, input("enter components of crack propagation direction (with space between each component):"). strip(). split()))
print(prop_array)
plane_array=list(map(int, input("enter components of crack plane direction (with space between each component):"). strip(). split()))
print(plane_array)

sur_ene=float(input('Enter surface energy of crack plane in units of J/m^2 \n'))  # surface energy of (100) is 1.177 J/m^2

x_prop_dir,y_prop_dir,z_prop_dir = prop_array[0], prop_array[1], prop_array[2]

x_plane_dir,y_plane_dir,z_plane_dir = plane_array[0], plane_array[1], plane_array[2]

S=get_rotated_compliances(x_prop_dir,y_prop_dir,z_prop_dir,x_plane_dir,y_plane_dir,z_plane_dir)

C=np.linalg.inv(S) 

'''
Solving eigenvalue problem presented in the article 'Andric P and Curtin W 2018 Atomistic modeling of fracture Modelling Simul. Mater. Sci. Eng. 27 013001'
'''

Q=np.array([[C[0,0],C[0,5],C[0,4]],[C[0,5],C[5,5],C[4,5]],[C[0,4],C[4,5],C[4,4]]])

R=np.array([[C[0,5],C[0,1],C[0,3]],[C[5,5],C[1,5],C[3,5]],[C[4,5],C[1,4],C[3,4]]])  

T=np.array([[C[5,5],C[1,5],C[3,5]],[C[1,5],C[1,1],C[1,3]],[C[3,5],C[1,3],C[3,3]]])

R_T=R.transpose()
N1=np.matmul(-inv(T),R_T)

N2=inv(T)

N3_1=np.matmul(R,inv(T))
N3=np.subtract(np.matmul(N3_1,R_T),Q)

N1_T=N1.transpose()
'''
print (N1.shape)
print (N2.shape)
print (N3.shape)
print (N1_T.shape)
'''

N=np.array([[N1[0,0],N1[0,1],N1[0,2],N2[0,0],N2[0,1],N2[0,2]],[N1[1,0],N1[1,1],N1[1,2],N2[1,0],N2[1,1],N2[1,2]],[N1[2,0],N1[2,1],N1[2,2],N2[2,0],N2[2,1],N2[2,2]],[N3[0,0],N3[0,1],N3[0,2],N1_T[0,0],N1_T[0,1],N1_T[0,2]],[N3[1,0],N3[1,1],N3[1,2],N1_T[1,0],N1_T[1,1],N1_T[1,2]],[N3[2,0],N3[2,1],N3[2,2],N1_T[2,0],N1_T[2,1],N1_T[2,2]]])
#print (N.shape)

#print (N)

print ("\nSolving eigenvalue problem has begun...")
##Eigenvalue problem solution: Eigenvalues and eigenvectors 
eigenval,eigenvec=la.eig(N)

#print ('Eigenvalues are: ',eigenval)  #This line prints eigenvalues
#print ('Eigenvectors are: ',eigenvec) # This line prints eigenvectors

# Eigenvalues and eigenvectors are saved to text files
np.savetxt('eigenvalues.dat',eigenval)
np.savetxt('eigenvectors.dat',eigenvec)

# Vector A is made up of eigenvectors: Details in above article.
A=np.array([[eigenvec[0,0],eigenvec[0,2],eigenvec[0,4]],[eigenvec[1,0],eigenvec[1,2],eigenvec[1,4]],[eigenvec[2,0],eigenvec[2,2],eigenvec[2,4]]])
print ("Eigenvector matrix A: ",A)

# Vector B is made up of eigenvectors: Details in above article.
B=np.array([[eigenvec[3,0],eigenvec[3,2],eigenvec[3,4]],[eigenvec[4,0],eigenvec[4,2],eigenvec[4,4]],[eigenvec[5,0],eigenvec[5,2],eigenvec[5,4]]])
print ("Eigenvector matrix B: ",B)

'''
print (inv(B))
print (np.matmul(B,inv(B)))

print (np.matmul(A,inv(B)))
print (np.multiply((np.matmul(A,inv(B))),1j))
print (np.multiply((np.matmul(A,inv(B))),1j).real)
'''
print (np.multiply((np.multiply((np.matmul(A,inv(B))),1j).real),0.5))
print ("\nSolving for Stroh's energy tensor '\u039B' ... ")
capital_lambda=np.multiply((np.multiply((np.matmul(A,inv(B))),1j).real),0.5)

Griffith_modeI_K=np.sqrt((2*sur_ene)/capital_lambda[1,1])/1e6
print ("Critical K for mode-I loading, according to Griffith's model is: ",Griffith_modeI_K,' MPa.m^{1/2}')
