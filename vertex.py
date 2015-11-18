# -*- coding: utf-8 -*-
import numpy as np
import math as m
import matplotlib.pyplot as plt
import random

##########################    Vertex Class     #################################

class Vertex:

    def __init__(self, alpha, rho0, k, n=100):
        #Initialize a vertex
        self.alpha, self.k, self.rho0=alpha, k, rho0                                          #Rest fold angles--not flat state, but rest state (radians)
        self.eps=2*np.pi-np.sum(alpha) if  np.abs(2*np.pi-np.sum(alpha))>1e-14 else 0         #Deviation from 2pi
        if alpha[1]<alpha[3]:
            alpha[1], alpha[3]=alpha[3], alpha[1]
        self.phi=np.asarray([np.sum(self.alpha[:i]) for i in range(4)])                       #absolute angles around vertex in flat state
        self.rf=np.vstack([np.cos(self.phi), np.sin(self.phi) , np.array([0]*4) ]).T
        self.getBA()                                                                          #calculate the binding angles on each branch
        self.gather(n=n)

    def getBA(self):
        #Get the binding angles of a vertex
        a=self.alpha
        self.bp=np.where((np.abs(self.alpha-np.roll(self.alpha, -1))-np.abs(np.roll(self.alpha, 2)-np.roll(self.alpha, 1)) > 0) & (np.abs(self.alpha-np.roll(self.alpha, 1))-np.abs(np.roll(self.alpha, -2)-np.roll(self.alpha, -1)) >0))[0][0]
        if self.bp == 0:
            self.brhoI=np.array([SA(a[2], a[3], a[1]-a[0], m=1), np.pi, np.pi-SA(a[3], a[1]-a[0], a[2], m=1), np.pi-SA(a[1]-a[0], a[2], a[3], m=1)])
            self.brhoII=np.array([np.pi, SA(a[2], a[3]-a[0], a[1], m=1), np.pi-SA(a[3]-a[0], a[1], a[2], m=1), np.pi-SA(a[1], a[2], a[3]-a[0], m=1)])
        else:
            self.brhoI=np.array([SA(a[1], a[0], a[2]-a[3], m=1), np.pi-SA(a[2]-a[3], a[0], a[1], m=1), np.pi-SA(a[0], a[1], a[2]-a[3], m=1), np.pi])
            self.brhoII=np.array([np.pi-SA(a[2]-a[1], a[0], a[3], m=1), SA(a[3], a[2]-a[1], a[0], m=1), np.pi, np.pi-SA(a[0], a[3], a[2]-a[1], m=1)])

    def clear(self):
        #Clear the folding/energy curves of a vertex
        self.rhoI, self.rhoII, self.eI, self.eII =[], [], [], []

################################################################################

    def getRho(self, rho0, branch):
        #Gather the folding curves for a vertex (positive inputs rho0 only)
        D24=np.arccos(np.cos(self.alpha[0])*np.cos(self.alpha[3])-np.sin(self.alpha[0])*np.sin(self.alpha[3])*np.cos(rho0))
        s1, s2, s4=SA(D24,self.alpha[0], self.alpha[3]), SA(self.alpha[3],self.alpha[0],D24), SA(self.alpha[0], D24, self.alpha[3])
        t2, t3, t4=SA(self.alpha[2], self.alpha[1], D24), SA(D24, self.alpha[1], self.alpha[2]), SA(self.alpha[1], self.alpha[2], D24)
        rhoI=np.array([s1, -np.pi+s2-t2, -t3, -np.pi+s4-t4])*np.sign(rho0)
        rhoII=np.array([s1, -np.pi+s2+t2, t3, -np.pi+s4+t4])*np.sign(rho0)
        if branch==0:
            return(np.transpose(rhoI))
        else:
            return(np.transpose(rhoII))

################################################################################

    def gather(self, n=100, delta=1e-6):
        #Gather the folding curves and energy curves of a vertex
        valsI, valsII = np.linspace(delta, self.brhoI[0]-delta, n/2), np.linspace(delta, self.brhoII[0]-delta, n/2)
        rhoI, rhoII=self.getRho(valsI, 0), self.getRho(valsII, 1)
        rhoI, rhoII=np.vstack([-rhoI[::-1,:], rhoI[:,:]]), np.vstack([-rhoII[::-1,:],  rhoII[:,:]])
        length=np.min([rhoI.shape[0], rhoII.shape[0]])
        rhoI, rhoII=rhoI[:length,:] , rhoII[:length,:]
        eI, eII=np.sum(self.k*(self.rho0-rhoI)**2, axis=1)+np.random.uniform(-1e-14, 1e-14, length), np.sum( self.k*(self.rho0-rhoII)**2, axis=1) +np.random.uniform(-1e-14, 1e-14, length)
        mrhoI=rhoI[np.r_[True, eI[1:] < eI[:-1]] & np.r_[eI[:-1] < eI[1:], True],:]
	mrhoII=rhoII[np.r_[True, eII[1:] < eII[:-1]] & np.r_[eII[:-1] < eII[1:], True],:]
        mrho=np.vstack([mrhoI, mrhoII])
        meI=eI[np.r_[True, eI[1:] < eI[:-1]] & np.r_[eI[:-1] < eI[1:], True]]
        meII=eII[np.r_[True, eII[1:] < eII[:-1]] & np.r_[eII[:-1] < eII[1:], True]]
        me=np.append(meI, meII)
        nm, nmI, nmII=me.size, meI.size, meII.size
        self.nm, self.nmI, self.nmII=nm, nmI, nmII
        self.rhoI, self.rhoII, self.eI, self.eII=rhoI, rhoII, eI, eII
        self.mrho, self.mrhoI, self.mrhoII, self.meI, self.meII, self.me=mrho, mrhoI, mrhoII, meI, meII, me

        #Now the curves and minima for below 2pi
        rhoBI=np.vstack([rhoI[np.sum(rhoI, 1)<0,:], rhoII[np.sum(rhoII, 1)<0,:]])
        eBI=np.append(eI[np.sum(rhoI, 1)<0], eII[np.sum(rhoII, 1)<0])
        rhoBII=np.vstack([rhoI[np.sum(rhoI, 1)>0,:], rhoII[np.sum(rhoII, 1)>0,:]])
        eBII=np.append(eI[np.sum(rhoI, 1)>0], eII[np.sum(rhoII, 1)>0])
        self.eBI=eBI[np.argsort(rhoBI[:,0])]
        self.rhoBI=rhoBI[np.argsort(rhoBI[:,0]),:]
        self.eBII=eBII[np.argsort(rhoBII[:,0])]
        self.rhoBII=rhoBII[np.argsort(rhoBII[:,0]),:]

        self.mrhoBI, self.meBI=mrho[np.sum(mrho, axis=1)<0,:], me[np.sum(mrho, axis=1)<0]
        self.mrhoBII, self.meBII=mrho[np.sum(mrho, axis=1)>0,:], me[np.sum(mrho, axis=1)>0]

        testRhoI, testRhoII = self.getRho(delta, 0), self.getRho(delta, 1)
        testRhoI, testRhoII = np.vstack([-testRhoI, testRhoI]), np.vstack([-testRhoII, testRhoII])

        testeI, testeII=np.sum(self.k*(self.rho0-testRhoI)**2, axis=1), np.sum( self.k*(self.rho0-testRhoII)**2, axis=1)

        if (np.sum(testRhoI[1,:])>0 and np.diff(testeI)<0 and np.diff(testeII)>0) or (np.sum(testRhoI[1,:])<0 and np.diff(testeI)>0 and np.diff(testeII)<0):
            self.mrhoBI=np.vstack([np.array([0,0,0,0]), self.mrhoBI])
            self.meBI=np.append(np.array(testeI[0]), self.meBI)
        if (np.sum(testRhoI[1,:])<0 and np.diff(testeI)<0 and np.diff(testeII)>0) or (np.sum(testRhoI[1,:])>0 and np.diff(testeI)>0 and np.diff(testeII)<0):
            self.mrhoBII=np.vstack([np.array([0,0,0,0]), self.mrhoBII])
            self.meBII=np.append(np.array(testeI[0]), self.meBII)

        #Get the above 2pi curves
        rhoAI=np.vstack([rhoI[rhoI[:,0] <= 0,:], rhoII[rhoII[:,0] <= 0,:]])
        eAI=np.append(eI[rhoI[:,0] <= 0], eII[rhoII[:,0] <= 0])
        rhoAII=np.vstack([rhoI[rhoI[:,0] >= 0,:], rhoII[rhoII[:,0] >= 0,:]])
        eAII=np.append(eI[rhoI[:,0] >= 0], eII[rhoII[:,0] >= 0])
        self.eAI=eAI[np.argsort(rhoAI[:,2])]
        self.rhoAI=rhoAI[np.argsort(rhoAI[:,2]),:]
        self.eAII=eAII[np.argsort(rhoAII[:,2])]
        self.rhoAII=rhoAII[np.argsort(rhoAII[:,2]),:]

        self.mrhoAI, self.meAI=mrho[mrho[:,0] <= 0,:], me[mrho[:,0] <= 0]
        self.mrhoAII, self.meAII=mrho[mrho[:,0] >= 0,:], me[mrho[:,0] >= 0]

        self.splitType = 2 if np.diff(testeI)*np.diff(testeII) > 0 else 1

        if (np.diff(testeI)>0 and np.diff(testeII)>0):
            self.mrhoAII=np.vstack([np.array([0,0,0,0]), self.mrhoAII])
            self.meAII=np.append(np.array(testeI[0]), self.meAII)
        if (np.diff(testeI)<0 and np.diff(testeII)<0):
            self.mrhoAI=np.vstack([np.array([0,0,0,0]), self.mrhoAI])
            self.meAI=np.append(np.array(testeI[0]), self.meAI)

        self.nmB, self.nmBI, self.nmBII=self.meBI.size+self.meBII.size, self.meBI.size, self.meBII.size
        self.nmA, self.nmAI, self.nmAII=self.meAI.size+self.meAII.size, self.meAI.size, self.meAII.size

        #Now determine the vector to represent the splitting topology
        if self.splitType == 1:
            if testeI[1] < testeII[1]:
                n_upleft = self.meI[self.mrhoI[:,0]<0].size
                n_lowright = self.meI[self.mrhoI[:,0]>0].size
                n_upright = self.meII[self.mrhoII[:,0]>0].size
                n_lowleft = self.meII[self.mrhoII[:,0]<0].size
            else:
                n_upleft = self.meII[self.mrhoII[:,0]<0].size
                n_lowright = self.meII[self.mrhoII[:,0]>0].size
                n_upright = self.meI[self.mrhoI[:,0]>0].size
                n_lowleft = self.meI[self.mrhoI[:,0]<0].size
        else:
            f = -1 if (np.diff(testeI)>0) else 1
            if testeI[1] < testeII[1]:
                n_upleft = self.meII[f*self.mrhoII[:,0]<0].size
                n_lowright = self.meII[f*self.mrhoII[:,0]>0].size
                n_upright = self.meI[f*self.mrhoI[:,0]<0].size
                n_lowleft = self.meI[f*self.mrhoI[:,0]>0].size
            else:
                n_upleft = self.meI[f*self.mrhoI[:,0]<0].size
                n_lowright = self.meI[f*self.mrhoI[:,0]>0].size
                n_upright = self.meII[f*self.mrhoII[:,0]<0].size
                n_lowleft = self.meII[f*self.mrhoII[:,0]>0].size

        if n_lowleft < n_lowright:
            n_lowleft, n_lowright = n_lowright, n_lowleft
            n_upleft, n_upright = n_upright, n_upleft
        if n_lowleft == n_lowright and n_upleft != n_upright:
            if n_upleft < n_upright:
                n_upleft, n_upright = n_upright, n_upleft
        self.splitVector = np.array([self.nm, self.splitType, n_lowleft, n_lowright, n_upleft, n_upright])
        self.splitString = ''.join(str(self.splitVector[i]) for i in np.arange(self.splitVector.size))

    def check(self):
        if np.isnan(np.concatenate([self.rhoI.flatten(), self.rhoII.flatten(), self.eI.flatten(), self.eII.flatten()])).any():
            return(0)
        else:
            return(1)

###################   Plotting Programs   ######################################
################################################################################

    def pfs(self):
        #Plot the flat state of a vertex
        plt.close('all')
        plt.figure(1, figsize=(5*1.2, 5*1.2))
        axf=plt.axes([0.05,0.05,0.9,0.9])
        plt.axis([-1.4*5, 1.4*5,-1.4*5, 1.4*5])
        plt.setp(axf.get_xticklabels(), visible=False)
        plt.setp(axf.get_yticklabels(), visible=False)
        axf.yaxis.set_tick_params(size=0)
        axf.xaxis.set_tick_params(size=0)
        dtheta=np.pi/50
        theta=np.arange(0, 2*np.pi-self.eps, dtheta)
        axf.scatter(5*np.cos(theta[theta<2*np.pi]),5*np.sin(theta[theta<2*np.pi]), c='black', marker='.', s=20 )
        axf.scatter(5*1.1*np.cos(theta[theta>2*np.pi]),5*1.1*np.sin(theta[theta>2*np.pi]), c='red', marker='.', s=20 )
        axf.plot( [0, 5*1.1*self.rf[0,0]], [0, 5*1.1*self.rf[0,1]], 'r--', linewidth=0.76  )
        axf.plot( [0, 5*1.1*np.cos(2*np.pi-self.eps)], [0, 5*1.1*np.sin(2*np.pi-self.eps)], 'k-.', alpha=0.65, linewidth=0.76  )
        for i in np.arange(1, 4):
            if self.phi[i]<2*np.pi:
                axf.plot( np.array([0, 5*self.rf[i,0]]), np.array([0, 5*self.rf[i,1]]), 'r--', linewidth=0.76  )
            else:
                axf.plot( np.array([0, 5*self.rf[i,0]]), np.array([0, 5*self.rf[i,1]]), 'r-.', linewidth=0.76  )

    def pspace(self, eps='f'):
        if eps=='f':
            rhoI, rhoII, eI, eII, mrhoI, mrhoII, meI, meII=self.rhoI, self.rhoII, self.eI, self.eII, self.mrhoI, self.mrhoII, self.meI, self.meII
        if eps=='b':
            rhoI, rhoII, eI, eII, mrhoI, mrhoII, meI, meII=self.rhoBI, self.rhoBII, self.eBI, self.eBII, self.mrhoBI, self.mrhoBII, self.meBI, self.meBII
        if eps=='a':
            rhoI, rhoII, eI, eII, mrhoI, mrhoII, meI, meII=self.rhoAI, self.rhoAII, self.eAI, self.eAII, self.mrhoAI, self.mrhoAII, self.meAI, self.meAII
        plt.close('all')
        fig1 = plt.figure()
        ax1, ax2, ax3, ax4=fig1.add_subplot(221), fig1.add_subplot(222, aspect='equal'), fig1.add_subplot(223, aspect='equal'), fig1.add_subplot(224, aspect='equal')
        ax1.set_ylim([0, np.nanmax(np.append(eI, eII))*1.2]); ax1.set_xlim([-np.pi, np.pi]); ax1.set_xlabel('rho_0'); ax1.set_ylabel('E')
        ax1.scatter(mrhoI[:,0], meI, facecolor='r', edgecolor='r', marker=(5, 1, 0), s=100, zorder=10)
        ax1.scatter(mrhoII[:,0], meII, facecolor='b', edgecolor='b', marker=(5, 1, 0), s=100, zorder=10)
        ax1.plot(rhoI[:,0], eI, 'r')
        ax1.plot(rhoII[:,0], eII, 'b')
        ax2.set_ylim([-np.pi, np.pi]); ax2.set_xlim([-np.pi, np.pi]); ax2.set_xlabel('rho_0'); ax2.set_ylabel('rho_1')
        ax2.plot(rhoI[:,0], rhoI[:,1], 'r')
        ax2.plot(rhoII[:,0], rhoII[:,1], 'b')
        ax3.set_ylim([-np.pi, np.pi]); ax3.set_xlim([-np.pi, np.pi]); ax3.set_xlabel('rho_0'); ax3.set_ylabel('rho_2')
        ax3.plot(rhoI[:,0], rhoI[:,2], 'r')
        ax3.plot(rhoII[:,0], rhoII[:,2], 'b')
        ax4.set_ylim([-np.pi, np.pi]); ax4.set_xlim([-np.pi, np.pi]); ax4.set_xlabel('rho_0'); ax4.set_ylabel('rho_3')
        ax4.plot(rhoI[:,0], rhoI[:,3], 'r')
        ax4.plot(rhoII[:,0], rhoII[:,3], 'b')

    def pEnergy(self):
        plt.close('all')
        fig1 = plt.figure()
        ax1, ax2, ax3=fig1.add_subplot(131), fig1.add_subplot(132), fig1.add_subplot(133)
        ax1.set_ylim([0, np.nanmax(np.append(self.eI, self.eII))*1.2]); ax1.set_xlim([-np.pi, np.pi]); ax1.set_xlabel('rho_0'); ax1.set_ylabel('Ef')
        ax1.scatter(self.mrhoI[:,0], self.meI, facecolor='r', edgecolor='r', marker=(5, 1, 0), s=100, zorder=10)
        ax1.scatter(self.mrhoII[:,0], self.meII, facecolor='b', edgecolor='b', marker=(5, 1, 0), s=100, zorder=10)
        ax1.plot(self.rhoI[:,0], self.eI, 'r'); ax1.plot(self.rhoII[:,0], self.eII, 'b')
        ax2.set_ylim([0, np.nanmax(np.append(self.eI, self.eII))*1.2]); ax1.set_xlim([-np.pi, np.pi]); ax1.set_xlabel('rho_0'); ax1.set_ylabel('Eb')
        ax2.scatter(self.mrhoBI[:,0], self.meBI, facecolor='r', edgecolor='r', marker=(5, 1, 0), s=100, zorder=10)
        ax2.scatter(self.mrhoBII[:,0], self.meBII, facecolor='b', edgecolor='b', marker=(5, 1, 0), s=100, zorder=10)
        ax2.plot(self.rhoBI[:,0], self.eBI, 'r'); ax2.plot(self.rhoBII[:,0], self.eBII, 'b')
        ax3.set_ylim([0, np.nanmax(np.append(self.eI, self.eII))*1.2]); ax1.set_xlim([-np.pi, np.pi]); ax1.set_xlabel('rho_0'); ax1.set_ylabel('Ea')
        ax3.scatter(self.mrhoAI[:,0], self.meAI, facecolor='r', edgecolor='r', marker=(5, 1, 0), s=100, zorder=10)
        ax3.scatter(self.mrhoAII[:,0], self.meAII, facecolor='b', edgecolor='b', marker=(5, 1, 0), s=100, zorder=10)
        ax3.plot(self.rhoAI[:,0], self.eAI, 'r'); ax3.plot(self.rhoAII[:,0], self.eAII, 'b')

################################################################################
############### Generating class objects (outside class) #######################
################################################################################

def genArb(eps=0, alpha=0, delta=1e-3):
    #Generate an arbitrary vertex with uniform probabilities over alpha space
    #Nonzero eps will create a vertex whose angles are sum(alpha)=2pi-eps
    #Inputting alpha will set alpha
    #Changing delta affects how close max/min angles approach pi/0 (or otherwise if eps nonzero)
    if np.size(alpha)==1:
        maxalpha, minalpha=10, 0
        while maxalpha > (np.pi-eps/2.)*(1-delta) or minalpha < np.pi*delta:
            alpha=uunifast(4, 2*np.pi-eps)
            maxalpha, minalpha=np.max(alpha), np.min(alpha)
        a=np.roll(alpha, -np.argmin(alpha))
        if a[1]<a[3]:
	   a[1], a[3]=a[3], a[1]
        aplus, aminus=a+np.roll(a,-1), a+np.roll(a,1)
        alpha=np.roll(a, -np.where((aminus < np.pi) & (aplus < np.pi))[0][0])
    else:
        aplus, aminus=alpha+np.roll(alpha,-1), alpha+np.roll(alpha,1)
        alpha=np.roll(alpha, -np.where((aminus < np.pi) & (aplus < np.pi))[0][0])
        eps=2*np.pi-np.sum(alpha)
    rho0=np.random.uniform(-np.pi, np.pi, 4)
    k=np.random.uniform(0, 1, 4)
    return(Vertex(alpha, rho0, k))

#############################   Tools  #########################################

def mrot(N,phi):
    #Returns the rotation matrix with angle phi and axis direction N (must be normed)
    A0 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    A1 = np.array([[N[0]*N[0],N[0]*N[1],N[0]*N[2]],
                   [N[1]*N[0],N[1]*N[1],N[1]*N[2]],
                   [N[2]*N[0],N[2]*N[1],N[2]*N[2]]])
    A2 = np.array([[0,-N[2],N[1]],
                   [N[2],0,-N[0]],
                   [-N[1],N[0],0]])
    return np.cos(phi)*A0 + (1-np.cos(phi))*A1 + np.sin(phi)*A2

def SA(a,b,c, m=0):
    #Returns variant of spherical law of cosines with optional sign change (m=1 or 0)
    if m==0:
        return(np.arccos((-np.cos(a)+np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c))))
    else:
        return(np.arccos((np.cos(a)-np.cos(b)*np.cos(c))/(np.sin(b)*np.sin(c))))

def uunifast(num_tasks, utilization):
    #Returns vector of length num_tasks whose sum is utilization with uniform sampling
    sum_u = utilization
    util = list()
    for i in range(1, num_tasks):
        next_sum_u = sum_u * m.pow(random.random(), 1.0 / float(num_tasks - i))
        util.append(sum_u - next_sum_u)
        sum_u = next_sum_u
    util.append(sum_u)
    return util
