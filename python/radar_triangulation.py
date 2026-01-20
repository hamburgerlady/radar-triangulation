#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 13:56:04 2025

@author: magnuso
"""

import numpy as np

def solver_radar_linear(yi,ri,ni):

    # Linear approximate solver for
    # return r, res 
  
    # Subtract mean sender positions.
    yim = np.mean(yi,1)
    yi2m = np.mean(np.sum(yi*yi,0))
    rim = np.mean(ri)
    A1 = 2*(yim-yi.T)
    b1 = -np.sum(yi*yi,0)+yi2m+(ri*ri)-(rim*rim)
    A2 = ni.T
    b2 = np.sum(ni*yi,0)
    A = np.vstack([A1,A2])
    b = np.hstack([b1,b2])   
    rsol = np.linalg.lstsq(A,b,rcond = None)    
    r = rsol[0]
    res =  A@r-b    
    return r,res



def solver_opt_radar(y0i,ri,ni,si,di,use_eigenvector = True):
    # Finds all stationary points to
    # min_x sum_j w(j) * (|x-s(:,k)|^2 - d(k)^2)^2
    # returns R, sols, res


    n = 3

    w = 1/(8*ri*ri*si*si)
    g = 1/(2*ri*ri*di*di)

    w = w.reshape(-1)
    g = g.reshape(-1)
    ri = ri.reshape(-1)
    nw = np.sum(w)
    w = w/nw
    g = g/nw
    # shift center
    t = np.sum(y0i*w,1)
    yi = (y0i.T-t).T    

    # Construct A,b such that (x'*x)*x + A*x + b = 0
  
    wy2mr2 = w*(np.sum(yi*yi,0).T-ri*ri)
    A = 2*(yi* w) @ yi.T + np.sum(wy2mr2) * np.eye(n) + 0.5*(ni* g) @ ni.T
    b = -yi@wy2mr2 -0.5*ni@(g*sum(ni*yi,0))
    eigv, V = np.linalg.eig(A)
    bb = V.T@b
    
    
    
    
    # basis = [x^2,y^2,z^2,x,y,z,1]
    AM = np.block([[-np.diag(eigv), np.diag(-bb), np.zeros((n,1))],[np.zeros((n,n)), -np.diag(eigv),-bb.reshape((n,1))],[np.ones((1,n)),np.zeros((1,n+1))]])
    
   
    
    if use_eigenvector:
        eigv2, VV = np.linalg.eig(AM)
        VV = VV / VV[-1,:]
        ro = V@VV[n:2*n,:]
    else:
        eigv2,_ = np.linalg.eig(AM)
        # eigenvector-less solution extraction
        ro = np.zeros((n,2*n+1),dtype = complex)
        z = np.vstack([np.zeros((n,n)),-np.eye(n)])
        for k in range(2*n+1):           
            T = AM - eigv2[k]*np.eye(2*n+1)
            lsqsol = np.linalg.lstsq(T[:,:-1].T,z,rcond = None)
            ro[:,k] = lsqsol[0].T@T[:,-1]
        ro = V@ro



    # perform some refinement on the roots
    for i in range(2*n+1):    
        roi = ro[:,i]
        if np.max(np.abs(roi.imag)) > 1e-6:
            continue
   
        for k in range(3):
            res = np.dot(roi,roi)*roi + A@roi + b
            if np.linalg.norm(res) < 1e-8:
                break
        
            J = np.sum(roi*roi)*np.eye(n) + 2 * np.outer(roi,roi) + A
            dx = np.linalg.lstsq(J,res, rcond = None)
            roi = roi - dx[0]
    
        ro[:,i] = roi

    
    # Revert translation of coordinate system
    sols = ro + t.reshape((3,1))
    
    

    # find best stationary point
    cost = np.inf
    R = np.zeros((n,1))
    for k in range(sols.shape[1]):
        if np.sum(np.abs(sols[:,k].imag)) > 1e-6:
            continue
        rok = sols[:,k].real  
        dist1 = (np.sqrt(np.sum((rok-y0i.T)*(rok-y0i.T),1))-ri)/si
        dist2 = (np.sum(ni.T*rok,1)-np.sum(ni*yi,0))/(ri*di)
        cost_k = np.sum(dist1*dist1) + np.sum(dist2*dist2)                      

        
        if cost_k < cost:
            cost = cost_k
            R = rok


    
    res = np.zeros((4,2*n+1),dtype=complex)
    for k in range(2*n+1):
        rok = ro[:,k]
        lambdak = eigv2[k]
        res[:3,k] = np.dot(rok,rok)*rok + A@rok + b
        res[-1,k] = lambdak - np.dot(rok,rok)
    return R, sols, res




def solver_opt_radar_prior(y0i,ri,ni,si,di,stz,use_eigenvector = True):
    # returns [R,sols,res]
    # Finds all stationary points to
    # min_x sum_j w(j) * (|x-s(:,k)|^2 - d(k)^2)^2

    n = 3

    h = 1/(stz*stz)
    w = 1/(8*ri*ri*si*si)
    g = 1/(2*ri*ri*di*di)

    w = w.reshape(-1)
    g = g.reshape(-1)
    ri = ri.reshape(-1)
    nw = np.sum(w)
    w = w/nw
    g = g/nw
    h = h/nw

    # shift center
    t = np.sum(y0i*w,1)
    yi = (y0i.T-t).T    

    # Construct A,b such that (x'*x)*x + A*x + b = 0
  
    wy2mr2 = w*(np.sum(yi*yi,0).T-ri*ri)
    A = 2*(yi* w) @ yi.T + np.sum(wy2mr2) * np.eye(n) + 0.5*(ni* g) @ ni.T
    b = -yi@wy2mr2 -0.5*ni@(g*sum(ni*yi,0))
    A = A+np.diag([0,0,h])
    b = b + h*np.array([0,0,t[-1]])

    
    
    
    eigv, V = np.linalg.eig(A)
    bb = V.T@b
    
    # basis = [x^2,y^2,z^2,x,y,z,1]
    AM = np.block([[-np.diag(eigv), np.diag(-bb), np.zeros((n,1))],[np.zeros((n,n)), -np.diag(eigv),-bb.reshape((n,1))],[np.ones((1,n)),np.zeros((1,n+1))]])

    if use_eigenvector:
        eigv2, VV = np.linalg.eig(AM)
        VV = VV / VV[-1,:]
        ro = V@VV[n:2*n,:]
    else:
        eigv2,_ = np.linalg.eig(AM)
        # eigenvector-less solution extraction
        ro = np.zeros((n,2*n+1),dtype = complex)
        z = np.vstack([np.zeros((n,n)),-np.eye(n)])
        for k in range(2*n+1):           
            T = AM - eigv2[k]*np.eye(2*n+1)
            lsqsol = np.linalg.lstsq(T[:,:-1].T,z,rcond = None)
            ro[:,k] = lsqsol[0].T@T[:,-1]
        ro = V@ro



    # perform some refinement on the roots
    for i in range(2*n+1):    
        roi = ro[:,i]
        if np.max(np.abs(roi.imag)) > 1e-6:
            continue
  
        for k in range(3):
            res = np.dot(roi,roi)*roi + A@roi + b
            if np.linalg.norm(res) < 1e-8:
                break
       
            J = np.sum(roi*roi)*np.eye(n) + 2 * np.outer(roi,roi) + A
            dx = np.linalg.lstsq(J,res, rcond = None)
            roi = roi - dx[0]
   
        ro[:,i] = roi


    # Revert translation of coordinate system
    sols = ro + t.reshape((3,1))

    # find best stationary point
    cost = np.inf
    R = np.zeros((n,1))
    for k in range(sols.shape[1]):
        if np.sum(np.abs(sols[:,k].imag)) > 1e-6:
            continue
        rok = sols[:,k].real
        dist1 = np.sum((rok-y0i.T)*(rok-y0i.T),1)- ri*ri
        dist2 = np.sum(ni.T*rok,1)-np.sum(ni*yi,0)
        cost_k = np.sum(w*dist1*dist1) + np.sum(g*dist2*dist2)
        if cost_k < cost:
            cost = cost_k
            R = rok


    res = np.zeros((4,2*n+1),dtype = complex)
    for k in range(2*n+1):
        rok = ro[:,k]
        lambdak = eigv2[k]
        res[:3,k] = np.dot(rok,rok)*rok + A@rok + b
        res[-1,k] = lambdak - np.dot(rok,rok)
    return R, sols, res


if __name__ == "__main__":
    # generate some random data
    np.set_printoptions(formatter={'float': lambda x: "{0:9.3f}".format(x)})
    
    sc = 100
    x0 = np.random.randn(3,1)*sc # 3D target point
    s0 = 0.1 # standard deviation of distance measurements
    d0 = 0.5/360*2*np.pi  # angle error
    N = 15  # number of radar positions 

    # random positions of radars and their measurements to the unknown x0
    yi = np.zeros((3,N)) 
    ni0 = np.zeros((3,N))

    for iii in range(N):
        kul = True
        while kul:
            R,_ = np.linalg.qr(np.random.randn(3,3))
            t = np.random.randn(3,1)*sc
            P = np.hstack([R,-R@t])
            u = P@np.vstack([x0,1])
            kul = u[-1]<0
    
        yi[:,iii] = t.flatten() # radar position
        v = np.cross(R[1,:],(x0-t).flatten()) 
        ni0[:,iii] = v/np.linalg.norm(v) # radar heading



    r0i = np.sqrt(np.sum((x0-yi)*(x0-yi),0))  # radar distance to x0
    ri = r0i+s0*np.random.randn(N) # radar distance measurement 

    ni = ni0+np.random.randn(3,N)*d0 # radar bearing measurement
    ni = ni/np.sqrt(np.sum(ni*ni,0))

    si = s0*np.ones(N) # stds of distances 
    di = d0*np.ones(N) # stds of bearings

    R0,sols0,res0 = solver_opt_radar(yi,r0i,ni0,si,di)  # ML solution without noise
    R,sols,res = solver_opt_radar(yi,ri,ni,si,di) # ML solution with noise
    R0lin,res0lin = solver_radar_linear(yi,r0i,ni0)  # Linear solution without noise
    Rlin,reslin = solver_radar_linear(yi,ri,ni) # Linear solution with noise

    x0 = x0.flatten()
    print('    GT    ML(no noise)    ML   Lin(no noise)  Lin :')
    print(np.vstack([x0,R0,R,R0lin,Rlin]).T)
    print('Distance to ground truth for ')
    print('     ML       Linear')
    print("{0:9.3f}".format(np.linalg.norm(x0-R)),"{0:9.3f}".format(np.linalg.norm(x0-Rlin)))

