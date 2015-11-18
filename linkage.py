# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os as os
import math as m
import random

class Linkage:
    
    def __init__(self, l, rho0, k, n=100):
        self.k, self.rho0=np.roll(k,-(np.argmin(l)+1)), np.roll(rho0, -(np.argmin(l)+1)) 
        self.l=np.roll(l, -(np.argmin(l)+1))/np.sum(l)                                                #Reorient l such that link 3 is shortest and longest=1
        self.Bl=np.where((np.abs(self.l-np.roll(self.l, -1))-np.abs(np.roll(self.l, 2)-np.roll(self.l, 1)) > 0) & (np.abs(self.l-np.roll(self.l, 1))-np.abs(np.roll(self.l, -2)-np.roll(self.l, -1)) >0))[0][0]
        self.grashof=1 if np.min(l)+np.max(l)<np.sum(l)-np.min(l)-np.max(l) else 0                #Determine whether or not linkage is grashof                   
        self.getIOAngles(n=n)
        self.getVectors()
        self.getRho()
        self.getE()

    def getIOAngles(self, n=100):
        delta=1e-7   #This ensures we don't calculate exactly on bounds and  get nans
        self.n=n                                                                                              #n is the number of points in the range -2pi -> 2pi
        g, b, h, a=self.l                                                                                     #recast l into standard linkage nomenclature (http://synthetica.eng.uci.edu/mechanicaldesign101/McCarthyNotes-2.pdf) 
        tMin=np.arccos(((h-b)**2-a**2-g**2)/(-2*a*g))*(1+1e-10)                                               #define tMin, tMax for input angle (as in website above)
        tMax=np.arccos(((h+b)**2-a**2-g**2)/(-2*a*g))*(1-1e-10)
        #t=np.linspace(-2*np.pi, 2*np.pi, n)                                                                   #create a theta array for input angle
        if np.isnan(tMin) and np.isnan(tMax):
            self.type='A'                                                                                     #if theta can continuously rotate
            t1=np.linspace(delta, np.pi+np.arccos((b**2-(h-a)**2-g**2)/(-2*(h-a)*g))-delta, n)
            A, B, C=(2*b*g-2*a*b*np.cos(t1)), -2*a*b*np.sin(t1), a**2+b**2+g**2-h**2-2*a*g*np.cos(t1)         
            phi1=np.arctan(B/A)+np.arccos(-C/np.sqrt(A**2+B**2))                                              
            t2, phi2=-t1, -phi1                                                                     
        if np.isnan(tMin) and not np.isnan(tMax):                                                             #if max exists but min doesn't
            self.type='B'
            tc1=np.arccos(((h+b)**2-a**2-g**2)/(-2*a*g))                          
            tc2=np.arccos((h**2-a**2-(g-b)**2)/(-2*a*(g-b)))
            tP, tM=np.linspace(delta, tc1-delta, np.round(n/2)), np.linspace(tc1-delta, tc2+delta, n-np.round(n/2)+1)[:-1]
            A, B, C=(2*b*g-2*a*b*np.cos(tP)), -2*a*b*np.sin(tP), a**2+b**2+g**2-h**2-2*a*g*np.cos(tP)            #readjust theta for non-intersecting linkage
            phiP=np.arctan(B/A)+np.arccos(-C/np.sqrt(A**2+B**2))
            A, B, C=(2*b*g-2*a*b*np.cos(tM)), -2*a*b*np.sin(tM), a**2+b**2+g**2-h**2-2*a*g*np.cos(tM)
            phiM=np.arctan(B/A)-np.arccos(-C/np.sqrt(A**2+B**2))
            t1, phi1=np.concatenate([tP, tM]), np.concatenate([phiP, phiM])
            t2, phi2=-t1, -phi1
        if not np.isnan(tMin) and np.isnan(tMax):                                                             #if min exists but max doesn't
            self.type='C'         
            if self.Bl==1:
                t1=np.linspace(tMin+delta, np.pi+np.arccos((h**2-a**2-(b-g)**2)/(-2*a*(b-g)))-delta, n)
                t2=np.linspace( np.pi-np.arccos((h**2-a**2-(b-g)**2)/(-2*a*(b-g)))+delta, 2*np.pi-tMin-delta, n)
            if self.Bl==2:
                t1=np.linspace(tMin+delta, np.pi+np.arccos((b**2-g**2-(h-a)**2)/(-2*g*(h-a)))-delta, n)
                t2=np.linspace(np.pi-np.arccos((b**2-g**2-(h-a)**2)/(-2*g*(h-a)))+delta, 2*np.pi-tMin-delta, n)
            A, B, C=(2*b*g-2*a*b*np.cos(t1)), -2*a*b*np.sin(t1), a**2+b**2+g**2-h**2-2*a*g*np.cos(t1)
            phi1=np.arctan(B/A)+np.arccos(-C/np.sqrt(A**2+B**2))
            A, B, C=(2*b*g-2*a*b*np.cos(t2)), -2*a*b*np.sin(t2), a**2+b**2+g**2-h**2-2*a*g*np.cos(t2)
            phi2=np.arctan(B/A)-np.arccos(-C/np.sqrt(A**2+B**2))   
        self.t1, self.t2, self.phi1, self.phi2=t1, t2, phi1, phi2 
        self.tMin, self.tMax=tMin, tMax
             
    def getVectors(self):
        P1=np.zeros([self.t1.size, 5, 2])
        P1[:,1,:]=np.array([np.repeat(self.l[0], self.t1.size), np.zeros(self.t1.size)]).T
        P1[:,2,:]=np.array([self.l[0]+self.l[1]*np.cos(self.phi1), self.l[1]*np.sin(self.phi1)]).T
        P1[:,3,:]=np.array([self.l[3]*np.cos(self.t1), self.l[3]*np.sin(self.t1)]).T
        P2=np.zeros([self.t2.size, 5, 2])
        P2[:,1,:]=np.array([np.repeat(self.l[0], self.t2.size), np.zeros(self.t2.size)]).T
        P2[:,2,:]=np.array([self.l[0]+self.l[1]*np.cos(self.phi2), self.l[1]*np.sin(self.phi2)]).T
        P2[:,3,:]=np.array([self.l[3]*np.cos(self.t2), self.l[3]*np.sin(self.t2)]).T
        self.P1, self.P2=P1, P2
        
    def draw(self, n, plot=1, outfile=0, branch=1):
        plt.close('all')
        if plot==0:
            matplotlib.pyplot.ioff()  
            matplotlib.use('Agg')
        fig = plt.figure(figsize=[5,5])
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim([np.min(self.P2[:,:,0])-0.1, np.max(self.P2[:,:,0])+0.1]); ax.set_ylim([np.min(self.P2[:,:,1])-0.1, np.max(self.P2[:,:,1])+0.1])
        if branch==1 or branch==3:
            ax.plot(self.P1[n,:,0], self.P1[n,:,1], 'k')
        if branch==2 or branch==3:
            ax.plot(self.P2[n,:,0], self.P2[n,:,1], 'r')
        if outfile != 0:
            plt.savefig(outfile)
        if plot==0:
            matplotlib.pyplot.ion()
                    
    def getRho(self):
        self.rho1=np.zeros([self.t1.size,4])
        self.rho2=np.zeros([self.t2.size,4])
        self.rho1[:,0]=sarcos(-self.P1[:,3,:], self.P1[:,1,:])
        self.rho1[:,1]=sarcos(self.P1[:,1,:], self.P1[:,2,:]-self.P1[:,1,:])
        self.rho1[:,2]=sarcos(self.P1[:,2,:]-self.P1[:,1,:], self.P1[:,3,:]-self.P1[:,2,:])
        self.rho1[:,3]=sarcos(self.P1[:,3,:]-self.P1[:,2,:], -self.P1[:,3,:])
        self.rho2[:,0]=sarcos(-self.P2[:,3,:], self.P2[:,1,:])
        self.rho2[:,1]=sarcos(self.P2[:,1,:], self.P2[:,2,:]-self.P2[:,1,:])
        self.rho2[:,2]=sarcos(self.P2[:,2,:]-self.P2[:,1,:], self.P2[:,3,:]-self.P2[:,2,:])
        self.rho2[:,3]=sarcos(self.P2[:,3,:]-self.P2[:,2,:], -self.P2[:,3,:])

    
    def getE(self):
        e1, e2=np.zeros(np.size(self.rho1[:,0])), np.zeros(np.size(self.rho2[:,0])) 
        self.e1, self.e2=np.sum(self.k*(self.rho0-self.rho1)**2, axis=1)+np.random.uniform(-1e-14, 1e-14, np.size(e1)), np.sum( self.k*(self.rho0-self.rho2)**2, axis=1) +np.random.uniform(-1e-14, 1e-14, np.size(e2))     
        self.mrho1=self.rho1[np.r_[True, self.e1[1:] < self.e1[:-1]] & np.r_[self.e1[:-1] < self.e1[1:], True],:]
        self.mrho2=self.rho2[np.r_[True, self.e2[1:] < self.e2[:-1]] & np.r_[self.e2[:-1] < self.e2[1:], True],:]
        self.mrho=np.vstack([self.mrho1, self.mrho2])
        self.me1=self.e1[np.r_[True, self.e1[1:] < self.e1[:-1]] & np.r_[self.e1[:-1] < self.e1[1:], True]]
        self.me2=self.e2[np.r_[True, self.e2[1:] < self.e2[:-1]] & np.r_[self.e2[:-1] < self.e2[1:], True]]
        self.me=np.append(self.me1, self.me2)
        
        self.nm1=self.me1.size
        self.nm2=self.me2.size

    def movie(self, branch=1):
        folder="/Users/ScottWaitukaitis/Desktop/linkageMovie/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filelist=[ f for f in os.listdir(folder) if f.endswith(".jpg") ]
        for f in filelist:
            os.remove(folder+f)
        for i in np.arange(self.t1.size):
            self.draw(i, plot=0, outfile=folder+'im'+str(i)+'.jpg', branch=branch)
            print i, "\r",
            
    def check(self):
        if np.isnan(np.concatenate([self.P1.flatten(), self.P2.flatten(), self.t1.flatten(), self.t2.flatten(), self.rho1.flatten(), self.rho2.flatten(), self.e1.flatten(), self.e2.flatten(), self.phi1.flatten(), self.phi2.flatten()])).any(): 
            return(0)
        else:
            return(1)
               
def sarcos(v1, v2):
    v1, v2=np.divide(v1.T, np.sqrt((v1*v1).sum(axis=1))).T, np.divide(v2.T, np.sqrt((v2*v2).sum(axis=1))).T
    return(np.arccos((v1*v2).sum(axis=1))*np.sign(np.cross(v1, v2))) 

def genArb(lengths=0, delta=1e-3):
    #Generate an arbitrary linkage with uniform probabilities over length space sum(lengths)=1
    #Inputting lengths will set lenghts
    #Changing delta affects how close max/min angles approach pi/0 (or otherwise if eps nonzero)
    if np.size(lengths)==1:
        maxlength, minlength=10, 0
        while maxlength > 0.5*(1-delta) or minlength < 0.5*delta:
            lengths=uunifast(4, 1)
            maxlength, minlength=np.max(lengths), np.min(lengths)
    rho0=np.random.uniform(-np.pi, np.pi, 4)
    k=np.random.uniform(0, 1, 4)                
    return(Linkage(lengths, rho0, k))  

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