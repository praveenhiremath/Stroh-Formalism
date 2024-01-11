from __future__ import division
import numpy as np
import io
import scipy
from array import *
import os
import math
from math import sqrt

'''
The code is for PhD research purposes

Author: Praveenkumar Hiremath
Email: praveenkumar.hiremath@mek.lth.se (Email at the University)
       praveenkumar.hiremath2911@gmail.com (Private email)
'''


'''
# Enter crack propagation and crack plane directions
#Crack propagation direction
o=-1
p=0
q=0
#o=np.array([0,0,4])

#Crack plane direction
u=0
v=1
w=0 
#u=np.array([4,1,0])

C_ij=np.loadtxt('./elastic.dat')   #,skiprows=0


#C=np.array([[195*1e9,107*1e9,113*1e9,0.0,0.0,0.0],[107*1e9,195*1e9,113*1e9,0.0,0.0,0.0],[113*1e9,113*1e9,213*1e9,0.0,0.0,0.0],[0.0,0.0,0.0,92*1e9,0.0,0.0],[0.0,0.0,0.0,0.0,92*1e9,0.0],[0.0,0.0,0.0,0.0,0.0,84*1e9]])
'''
def get_rotated_compliances(x_crack_prop_dir,y_crack_prop_dir,z_crack_prop_dir,x_crack_plane_dir,y_crack_plane_dir,z_crack_plane_dir):

    C_ij=np.loadtxt('./elastic.dat')   #,skiprows=0

    o, p, q = x_crack_prop_dir,y_crack_prop_dir,z_crack_prop_dir
    u, v, w = x_crack_plane_dir,z_crack_plane_dir,y_crack_plane_dir


    C=C_ij

    S_New=np.zeros([3,3,3,3])
    T=np.zeros([3,3])
    S_Old=np.zeros([3,3,3,3])

    S=np.linalg.inv(C)


#Building 4th order compliance tensor in original system --> (*) condition in Report.
    # Set 1
    S_Old[0,0,0,0]=S[0,0]   #s11
    S_Old[0,0,1,1]=S[0,1]   #s12
    S_Old[0,0,2,2]=S[0,2]   #s13

    # Set 2
    S_Old[0,0,0,1]=S[0,5]/2   #s16/2     
    S_Old[0,0,1,0]=S[0,5]/2   #s16/2

    # Set 3
    S_Old[0,1,0,1]=S[5,5]/4   #s66/4     
    S_Old[0,1,1,0]=S[5,5]/4   #s66/4    
    S_Old[1,0,0,1]=S[5,5]/4   #s66/4    
    S_Old[1,0,1,0]=S[5,5]/4   #s66/4

    # Set 4
    S_Old[0,1,1,1]=S[5,1]/2   #s62/2     
    S_Old[1,0,1,1]=S[5,1]/2   #s62/2

    # Set 5
    S_Old[0,1,2,2]=S[5,2]/2   #s63/2     
    S_Old[1,0,2,2]=S[5,2]/2   #s63/2

    # Set 6
    S_Old[1,1,1,1]=S[1,1]   #s22
    S_Old[1,1,2,2]=S[1,2]   #s23

    # Set 7
    S_Old[1,1,0,1]=S[1,5]/2   #s26/2     
    S_Old[1,1,1,0]=S[1,5]/2   #s26/2

    # Set 8
    S_Old[2,2,2,2]=S[2,2]   #s33
    S_Old[1,1,0,0]=S[1,0]   #s21
    S_Old[2,2,0,0]=S[2,0]   #s31
    S_Old[2,2,1,1]=S[2,1]   #s32

    # Set 9
    S_Old[1,2,1,2]=S[3,3]/4   #s44/4     
    S_Old[2,1,2,1]=S[3,3]/4   #s44/4
    S_Old[1,2,2,1]=S[3,3]/4   #s44/4
    S_Old[2,1,1,2]=S[3,3]/4   #s44/4

    # Set 10
    S_Old[0,2,0,2]=S[4,4]/4   #s55/4
    S_Old[2,0,2,0]=S[4,4]/4   #s55/4
    S_Old[0,2,2,0]=S[4,4]/4   #s55/4
    S_Old[2,0,0,2]=S[4,4]/4   #s55/4


    a=(p*w)-(q*v)
    b=(u*q)-(w*o)
    c=(o*v)-(p*u)

#Normalization of the above vectors to form Orthonormal basis set:
#Normalizing crack propagation direction
    X1=o/sqrt(pow(o,2)+pow(p,2)+pow(q,2))
    Y1=p/sqrt(pow(o,2)+pow(p,2)+pow(q,2))
    Z1=q/sqrt(pow(o,2)+pow(p,2)+pow(q,2))

#Normalizing crack front direction
    X3=a/sqrt(pow(a,2)+pow(b,2)+pow(c,2))
    Y3=b/sqrt(pow(a,2)+pow(b,2)+pow(c,2))
    Z3=c/sqrt(pow(a,2)+pow(b,2)+pow(c,2))

#Normalizing crack plane direction
    X2=u/sqrt(pow(u,2)+pow(v,2)+pow(w,2))
    Y2=v/sqrt(pow(u,2)+pow(v,2)+pow(w,2))
    Z2=w/sqrt(pow(u,2)+pow(v,2)+pow(w,2))

#Rotation matrix: T(ij)=x'(i).x(j) Here i is associated with the rotated system axes and j with original.
    T=np.array([[X1,Y1,Z1],[X2,Y2,Z2],[X3,Y3,Z3]])


# ROTATION OPERATION
    s=0; t=0; k=0; l=0;
#compliance constants in rotated coordinate system
    for s in range(0,3,1):
     for t in range(0,3,1):
      for k in range(0,3,1):
       for l in range(0,3,1):
        for g in range(0,3,1):
         for h in range(0,3,1):
          for m in range(0,3,1):
           for n in range(0,3,1):
             S_New[s,t,k,l]= S_New[s,t,k,l]+T[s,g]*T[t,h]*S_Old[g,h,m,n]*T[k,m]*T[l,n]  #Rotation

#No plane stress and No plane strain   (*) condition in Report.
    sN11=S_New[0,0,0,0]
    sN12=S_New[0,0,1,1]
    sN13=S_New[0,0,2,2]
    sN14=2*S_New[0,0,1,2]
    sN15=2*S_New[0,0,0,2]
    sN16=2*S_New[0,0,0,1]

    sN21=S_New[1,1,0,0]
    sN22=S_New[1,1,1,1]
    sN23=S_New[1,1,2,2]
    sN24=2*S_New[1,1,1,2]
    sN25=2*S_New[1,1,0,2]
    sN26=2*S_New[1,1,0,1]

    sN31=S_New[2,2,0,0]
    sN32=S_New[2,2,1,1]
    sN33=S_New[2,2,2,2]
    sN34=2*S_New[2,2,1,2]
    sN35=2*S_New[2,2,0,2]
    sN36=2*S_New[2,2,0,1]

    sN41=2*S_New[1,2,0,0]
    sN42=2*S_New[1,2,1,1]
    sN43=2*S_New[1,2,2,2]
    sN44=4*S_New[1,2,1,2]
    sN45=4*S_New[1,2,0,2]
    sN46=4*S_New[1,2,0,1]

    sN51=2*S_New[0,2,0,0]
    sN52=2*S_New[0,2,1,1]
    sN53=2*S_New[0,2,2,2]
    sN54=4*S_New[0,2,1,2]
    sN55=4*S_New[0,2,0,2]
    sN56=4*S_New[0,2,0,1]

    sN61=2*S_New[0,1,0,0]
    sN62=2*S_New[0,1,1,1]
    sN63=2*S_New[0,1,2,2]
    sN64=4*S_New[0,1,1,2]
    sN65=4*S_New[0,1,0,2]
    sN66=4*S_New[0,1,0,1]



#after rotation compliance constants to array
    Final_S=np.zeros([6,6])
    Final_S[0,0]=sN11; Final_S[0,1]=sN12; Final_S[0,2]=sN13; Final_S[0,3]=sN14; Final_S[0,4]=sN15; Final_S[0,5]=sN16; 
    Final_S[1,0]=sN21; Final_S[1,1]=sN22; Final_S[1,2]=sN23; Final_S[1,3]=sN24; Final_S[1,4]=sN25; Final_S[1,5]=sN26; 
    Final_S[2,0]=sN31; Final_S[2,1]=sN32; Final_S[2,2]=sN33; Final_S[2,3]=sN34; Final_S[2,4]=sN35; Final_S[2,5]=sN36; 
    Final_S[3,0]=sN41; Final_S[3,1]=sN42; Final_S[3,2]=sN43; Final_S[3,3]=sN44; Final_S[3,4]=sN45; Final_S[3,5]=sN46; 
    Final_S[4,0]=sN51; Final_S[4,1]=sN52; Final_S[4,2]=sN53; Final_S[4,3]=sN54; Final_S[4,4]=sN55; Final_S[4,5]=sN56; 
    Final_S[5,0]=sN61; Final_S[5,1]=sN62; Final_S[5,2]=sN63; Final_S[5,3]=sN64; Final_S[5,4]=sN65; Final_S[5,5]=sN66; 


    np.savetxt('Aug20_rotated_compliance.dat',Final_S)
    return Final_S




