#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:50:04 2018

@author: enrique, ariel
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import itertools

plt.close('all')

#--- model and pontryagin equations ---#

@jit
def derivadax1(x1,x2,x3,h,a1,b1,s1,a2,b2,s2,g1,g2,u,p1):
    k11 = a1*x1*(x2)**(g1) - (b1 + p1*u)*x1 + s1*x1*x3
    return k11

@jit
def derivadax2(x1,x2,x3,h,a2,b2,s2,g2,u,p2):
    k12 = a2*((x1)**(g2))*x2 - (b2 + p2*u)*x2 + s2*x2*x3
    return k12

@jit
def derivadax3(x1,x2,x3,h,a3,b3,k,g1,g2,s3,s4,u):
     k13 = a3*(x3*(1 - x3/k))- (b3 + u)*x3 + s3*(x1**g2)*x3 + s4*(x2**g1)*x3
     return k13
 
@jit
def adjunta1(x1,x2,x3,a1,b1,s1,a2,g1,g2,s3,p1,u,lamda1,lamda2,lamda3):
    k11 = -a2*g2*lamda2*x2*(x1)**(g2-1) - s3*g2*lamda3*((x1)**(g2-1))*x3-(a1*(x2)**(g1)+s1*x3-b1-p1*u)*lamda1
    return k11

@jit
def adjunta2(x1,x2,x3,a1,a2,b2,s2,g1,g2,s4,p2,u,lamda1,lamda2,lamda3):
    k12 = -a1*g1*lamda1*x1*(x2)**(g1-1) - s4*g1*lamda3*((x2)**(g1-1))*x3 - (a2*((x1)**(g2))+s2*x3-b2-p2*u)*lamda2
    return k12

@jit
def adjunta3(x1,x2,x3,s1,s2,a3,b3,k,g1,g2,s3,s4,u,lamda1,lamda2,lamda3):
    k13 = -s1*lamda1*x1 -s2*lamda2*x2 - (s3*(x1)**(g2) +s4*(x2)**(g1)-a3*(x3/k -1)-b3-u-(a3*x3)/k)*lamda3 -2*x3
    return k13

@jit 
def recta(t1,t2,y1,y2,t):
    return ((y2-y1)/(t2-t1))*(t1-t)+y1

@jit
def rungekuttaforward(t,N,x1,x2,x3,h,a1,b1,s1,a2,b2,s2,a3,b3,k,g1,g2,s3,s4,u,p1,p2):
    f_interp = InterpolatedUnivariateSpline(t,u, k=5)
    for i in range(0,N):
        k11 = derivadax1(x1[i],x2[i],x3[i],h,a1,b1,s1,a2,b2,s2,g1,g2,u[i],p1)
        k12 = derivadax2(x1[i],x2[i],x3[i],h,a2,b2,s2,g2,u[i],p2)
        k13 = derivadax3(x1[i],x2[i],x3[i],h,a3,b3,k,g1,g2,s3,s4,u[i])
        
        k21 = derivadax1(x1[i]+k11*h*0.25,
                         x2[i]+k12*h*0.25,
                         x3[i]+k13*h*0.25,
                         h,a1,b1,s1,a2,b2,s2,g1,g2,
                         f_interp(t[i] + h*0.25),
                         p1)
        
        k22 = derivadax2(x1[i]+k11*h*0.25,
                         x2[i]+k12*h*0.25,
                         x3[i]+k13*h*0.25,
                         h,a2,b2,s2,g2,
                         f_interp(t[i] + h*0.25),
                         p2)
        
        k23 = derivadax3(x1[i]+k11*h*0.25,
                         x2[i]+k12*h*0.25,
                         x3[i]+k13*h*0.25,
                         h,a3,b3,k,g1,g2,s3,s4,
                         f_interp(t[i] + h*0.25))
        
        k31 = derivadax1(x1[i]+k21*h*0.5,
                         x2[i]+k22*h*0.5,
                         x3[i]+k23*h*0.5,
                         h,a1,b1,s1,a2,b2,s2,g1,g2,
                         f_interp(t[i] + h*0.5),
                         p1)
        
        k32 = derivadax2(x1[i]+k21*h*0.5,
                         x2[i]+k22*h*0.5,
                         x3[i]+k23*h*0.5,
                         h,a2,b2,s2,g2,
                         f_interp(t[i] + h*0.5),
                         p2)
        
        k33 = derivadax3(x1[i]+ k21*h*0.5,
                         x2[i]+k22*h*0.5,
                         x3[i]+k23*h*0.5,
                         h,a3,b3,k,g1,g2,s3,s4,
                         f_interp(t[i] + h*0.5))
        
        k41 = derivadax1(x1[i]+k11*h-2*k21*h+2*k31*h,
                         x2[i]+k12*h-2*k22*h+2*k32*h,
                         x3[i]+k13*h-2*k23*h+2*k33*h,
                         h,a1,b1,s1,a2,b2,s2,g1,g2,
                         f_interp(t[i] + h),
                         p1)
        
        k42 = derivadax2(x1[i]+k11*h-2*k21*h+2*k31*h,
                         x2[i]+k12*h-2*k22*h+2*k32*h,
                         x3[i]+k13*h-2*k23*h+2*k33*h,
                         h,a2,b2,s2,g2,
                         f_interp(t[i] + h),
                         p2)
        
        k43 = derivadax3(x1[i]+k11*h -2.*k21*h +2.*k31*h,
                         x2[i]+k12*h -2.*k22*h +2.*k32*h,
                         x3[i]+k13*h -2.*k23*h +2.*k33*h,
                         h,a3,b3,k,g1,g2,s3,s4,
                         f_interp(t[i] + h))
        
        x1[i+1] = x1[i] + (h/6.)*(k11  + 4.*k31 + k41)
        x2[i+1] = x2[i] + (h/6.)*(k12  + 4.*k32 + k42)
        x3[i+1] = x3[i] + (h/6.)*(k13  + 4.*k33 + k43)
        
@jit
def rungekuttabackward(t,N,x1,x2,x3,h,a1,b1,s1,a2,b2,s2,a3,b3,k,g1,g2,s3,s4,p1,p2,u,lamda1,lamda2,lamda3):
    f_interp = InterpolatedUnivariateSpline(t,u,k=5)
    f_interpx1 = InterpolatedUnivariateSpline(t,x1, k=3)
    f_interpx2 = InterpolatedUnivariateSpline(t,x2, k=3)
    f_interpx3 = InterpolatedUnivariateSpline(t,x3, k=3)
    for i in range(N,0,-1):
        k11 =  adjunta1(x1[i],x2[i],x3[i],a1,b1,s1,a2,g1,g2,s3,p1,u[i],lamda1[i],lamda2[i],lamda3[i])
        k12 =  adjunta2(x1[i],x2[i],x3[i],a1,a2,b2,s2,g1,g2,s4,p2,u[i],lamda1[i],lamda2[i],lamda3[i])
        k13 =  adjunta3(x1[i],x2[i],x3[i],s1,s2,a3,b3,k,g1,g2,s3,s4,u[i],lamda1[i],lamda2[i],lamda3[i])
        
        k21 =  adjunta1(f_interpx1(t[i] - h*0.25),
                        f_interpx2(t[i] - h*0.25),
                        f_interpx3(t[i] - h*0.25),
                        a1,b1,s1,a2,g1,g2,s3,p1,
                        f_interp(t[i] - h*0.25),
                        lamda1[i] -k11*h*0.25,
                        lamda2[i] -k12*h*0.25,
                        lamda3[i] -k13*h*0.25)
        
        k22 = adjunta2(f_interpx1(t[i] - h*0.25),
                       f_interpx2(t[i] - h*0.25),
                       f_interpx3(t[i] - h*0.25),
                       a1,a2,b2,s2,g1,g2,s4,p2,
                       f_interp(t[i] - h*0.25),
                       lamda1[i] -k11*h*0.25,
                       lamda2[i] -k12*h*0.25,
                       lamda3[i] -k13*h*0.25)
        
        k23 =  adjunta3(f_interpx1(t[i] - h*0.25),
                        f_interpx2(t[i] - h*0.25),
                        f_interpx3(t[i] - h*0.25),
                        s1,s2,a3,b3,k,g1,g2,s3,s4,
                        f_interp(t[i] - h*0.25),
                        lamda1[i] -k11*h*0.25,
                        lamda2[i] -k12*h*0.25,
                        lamda3[i] -k13*h*0.25)
        
        k31 =  adjunta1(f_interpx1(t[i] - h*0.5),
                        f_interpx2(t[i] - h*0.5),
                        f_interpx3(t[i] - h*0.5),
                        a1,b1,s1,a2,g1,g2,s3,p1,
                        f_interp(t[i] - h*0.5),
                        lamda1[i] -k21*h*0.5,
                        lamda2[i] -k22*h*0.5,
                        lamda3[i] -k23*h*0.5)
       
        k32 = adjunta2(f_interpx1(t[i] - h*0.5),
                       f_interpx2(t[i] - h*0.5),
                       f_interpx3(t[i] - h*0.5),
                       a1,a2,b2,s2,g1,g2,s4,p2,
                       f_interp(t[i] - h*0.5),
                       lamda1[i] -k21*h*0.5,
                       lamda2[i] -k22*h*0.5,
                       lamda3[i] -k23*h*0.5)
        
        k33 = adjunta3(f_interpx1(t[i] - h*0.5),
                        f_interpx2(t[i] - h*0.5),
                        f_interpx3(t[i] - h*0.5),
                        s1,s2,a3,b3,k,g1,g2,s3,s4,
                        f_interp(t[i] - h*0.5),
                        lamda1[i] -k21*h*0.5,
                        lamda2[i] -k22*h*0.5,
                        lamda3[i] -k23*h*0.5)
        
        k41 = adjunta1(f_interpx1(t[i-1]  - h),
                        f_interpx2(t[i-1] - h),
                        f_interpx3(t[i-1] - h),
                        a1,b1,s1,a2,g1,g2,s3,p1,
                        f_interp(t[i] - h),
                        lamda1[i] -(h*k11 -2*h*k21 + 2*k31*h),
                        lamda2[i] -(h*k11 -2*h*k21 + 2*k31*h),
                        lamda3[i] -(h*k11 -2*h*k21 + 2*k31*h))
        
        k42 = adjunta2(f_interpx1(t[i-1]  - h),
                        f_interpx2(t[i-1] - h),
                        f_interpx3(t[i-1] - h),
                       a1,a2,b2,s2,g1,g2,s4,p2,
                       f_interp(t[i] - h),
                       lamda1[i] -(h*k11 -2.*h*k21 + 2.*k31*h),
                       lamda2[i] -(h*k11 -2.*h*k21 + 2.*k31*h),
                       lamda3[i] -(h*k11 -2.*h*k21 + 2.*k31*h))
        
        k43 = adjunta3(f_interpx1(t[i-1] - h),
                        f_interpx2(t[i-1] - h),
                        f_interpx3(t[i-1] - h),
                        s1,s2,a3,b3,k,g1,g2,s3,s4,
                        f_interp(t[i] - h),
                        lamda1[i] -(h*k11 -2.*h*k21 + 2.*k31*h),
                        lamda2[i] -(h*k11 -2.*h*k21 + 2.*k31*h),
                        lamda3[i] -(h*k11 -2.*h*k21 + 2.*k31*h))
        
        lamda1[i-1] = lamda1[i] - (h/6.)*(k11  +4.*k31 + k41)
        lamda2[i-1] = lamda2[i] - (h/6.)*(k12  +4.*k32 + k42)
        lamda3[i-1] = lamda3[i] - (h/6.)*(k13  +4.*k33 + k43)

#--- control update ---#
        
@jit    
def actualizarcontrol_singular(delta,t,N,x1,x2,x3,lamda1,lamda2,lamda3,p1,p2,wr,maxu,minu):
    u = np.zeros(N+1)
    
    conmutador = InterpolatedUnivariateSpline(t, wr - p1*x1*lamda1 - p2*x2*lamda2 - x3*lamda3, k=5)
    
    for i in range(0,N+1):
        if conmutador(t[i]) < 0.:
            u[i] = maxu
        else:
            u[i]= minu
    return u

def bang_simulation_Sc1(t0, tf, NoWin, B, MinU, MaxU):
    #--- sub-intervals creation ---#

    # sub-intervals number
    ventanas = NoWin

    # initial and final times
    a = float(t0)
    b = float(tf)

    # sub-intervals width
    dias = (b-a)/float(ventanas)

    # sub-intervals discretization
    N = int(dias)*10;

    #--- model and pontryagin parameters ---#

    x1 = np.zeros(N+1)
    x2 = np.zeros(N+1)
    x3 = np.zeros(N+1)

    lamda1 = np.zeros(N+1)
    lamda2 = np.zeros(N+1)
    lamda3 = np.zeros(N+1)

    x1[0]=4.42*10**(-6)
    x2[0]=4.46
    x3[0]=1000.0

    k = 10.0**(4)

    p1 = 1.0
    p2 = 1.0

    #- Scenario 1 -#

    a1 = 0.5
    a2 = 0.05
    a3 = 1.5*10**(-2)

    b1 = 0.2
    b2 = 0.02
    b3 = 0.0

    g1 = -0.3;
    g2 = 0.7

    s1 = 1.0*10**(-6)
    s2 = 0.0
    s3 = 1*10**(-3)
    s4 = 0*10**(-4)

    #- pontryagin -#

    wr   = B

    minu = MinU
    maxu = MaxU

    #--- forward-backward ---#

    delta  = 0.*10**(-7)
    delta2 = 1.*10**(-7)

    exponente = 0.5

    iteracion   = 0
    iteraciomax = 50

    JJ = []

    uu = []

    xx1 = []
    xx2 = []
    xx3 = []

    tt = []

    # 0 -> convergence, 1 -> no convergence
    flags = np.zeros(ventanas)

    for i in range(0,ventanas):
        print("Iteracion externa: ",i)
        aAux = a + i*dias
        bAux = a + (i+1)*dias

        t = np.linspace(aAux,bAux,N+1);
        h = dias/float(N)
        u = np.zeros(N+1)

        test = -1

        iteracion = 0
        #c = c**exponente

        J = [];
        while test < 0.0 :
            print("Iteracion interna: ",iteracion)

            oldu  = u.copy()

            oldx1 = x1.copy()
            oldx2 = x2.copy()
            oldx3 = x3.copy()

            oldlamda1 = lamda1.copy()
            oldlamda2 = lamda2.copy()
            oldlamda3 = lamda3.copy()

            rungekuttaforward(t,N,
                              x1,x2,x3,
                              h,a1,b1,s1,a2,b2,s2,a3,b3,k,g1,g2,s3,s4,u,p1,p2)

            rungekuttabackward(t,N,
                               x1,x2,x3,
                               h,a1,b1,s1,a2,b2,s2,a3,b3,k,g1,g2,s3,s4,p1,p2,u,
                               lamda1,lamda2,lamda3)

            conmutador = InterpolatedUnivariateSpline(t, wr - p1*x1*lamda1 - p2*x2*lamda2 - x3*lamda3, k=5)

            u = actualizarcontrol_singular(delta,t,N,
                                           x1,x2,x3,
                                           lamda1,lamda2,lamda3,
                                           p1,p2,wr,maxu,minu)

            J.append(np.trapz(wr*u + x3**2,t))

            temp1 = delta2*sum(abs(u)) -sum(abs(oldu -u))

            temp2 = delta2*sum(abs(x1)) -sum(abs(oldx1 -x1))
            temp3 = delta2*sum(abs(x2)) -sum(abs(oldx2 -x2))
            temp4 = delta2*sum(abs(x3)) -sum(abs(oldx3 -x3))

            temp5 = delta2*sum(abs(lamda1)) -sum(abs(oldlamda1 -lamda1))
            temp6 = delta2*sum(abs(lamda2)) -sum(abs(oldlamda2 -lamda2))
            temp7 = delta2*sum(abs(lamda3)) -sum(abs(oldlamda3 -lamda3))

            test = min(temp1,temp2,temp3,temp4,temp5,temp6,temp7)

            print(test)
            print(temp1)

            iteracion = iteracion + 1

            if(iteracion > iteraciomax):
                print("maximo numero de iteraciones")
                flags[i] = 1 #guardar la falta de convergencia
                break

        JJ.append(J[-1])

        uu.append(u)

        xx1.append(x1.copy())
        xx2.append(x2.copy())
        xx3.append(x3.copy())

        tt.append(t.copy())

        x1[0] = x1[N]
        x2[0] = x2[N]
        x3[0] = x3[N]

        lamda1 = np.zeros(N+1)
        lamda2 = np.zeros(N+1)
        lamda3 = np.zeros(N+1)

    #--- resultados ---#

    # join arrays
    xx1new = list(itertools.chain(*xx1))
    xx2new = list(itertools.chain(*xx2))
    xx3new = list(itertools.chain(*xx3))

    uuNew = list(itertools.chain(*uu))

    ttNew = list(itertools.chain(*tt))
    
    return ttNew, xx1new, xx2new, xx3new, uuNew

ttNew, xx1new, xx2new, xx3new, uuNew = bang_simulationSc1(0, 250, 25, 0., 0.01)

#--- plots ---#
plt.figure()
plt.plot(ttNew,xx1new)

plt.figure()
plt.plot(ttNew,xx2new)

plt.figure()
plt.plot(ttNew,xx3new)

plt.figure()
plt.plot(ttNew,uuNew)

print("#--REPORTE---#")
      
if any(flags==1):
    print("Algun sub-intervalo no convergio. Ver flags:")
    print(flags)
    
print("La integral de J vale: ",sum(JJ))
