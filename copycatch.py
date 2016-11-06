# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:39:27 2016

@author: Syed Mohsin Bukhari
"""

import numpy as np
import pickle
import ast

class CopyCatch:

    def __init__(self, n = 2, m = 2, dt = 1.0, phi = 1.0):
        """Initiates CopyCatch for n users who have liked phi*m pages in
        dt time.\n

        Keyword arguments:\n
        n -- the number of users (default 2)\n
        m -- the number of pages (default 2)\n
        dt -- the width of the time window in which perpetrators like pages\n
        phi -- the fraction of total number of pages that perpetrators like\n
        """
        #beta -- the factor by which to loosen the time window to adjust users
        #I -- the adjacency matrix which tells if a user has liked a page
        #L -- the data matrix which tells the time at which users liked pages
        #c -- the cluster center
        #U -- the list of all users
        #P -- the list of all pages
        #U_ -- the list of all suspected users
        #P_ -- the list of all suspected pages

        print('Initiated CopyCatch with n=' + str(n) + ', m=' + str(m) + \
            ', dt=' + str(dt) + ' and phi=' + str(phi) )

        self.n = n
        self.m = m
        self.dt = dt
        self.phi = phi

        self.beta = 2.0
        
#        self.L = np.matrix(\
#            [\
#            [0.0, 0.0, 0.0, 0.0, 5.2],\
#            [0.0, 6.1, 0.0, 9.1, 5.0],\
#            [0.0, 0.0, 7.9, 0.0, 5.1],\
#            [0.0, 5.7, 0.0, 9.3, 5.1],\
#            [3.1, 0.0, 0.0, 0.0, 0.0]] )

        self.L = self.readserializedmatrix('/home/elcid/workspace/data/maryam_nawaz_twitter/data.pkl')

        self.I = self.L > 0.0
        self.I = self.I.astype(int)

        self.U = set(range(0, self.L.shape[0] ))
        self.P = set(range(0, self.L.shape[1] ))

        #these two should be random
        #self.c = np.array([0.0, 5.0, 0.0, 9.0, 5.0])
        self.c = np.zeros(self.L.shape[1], dtype=np.float64)
        self.c[145]=1474153292000.0
        self.c[150]=1456513046000.0
#        self.P_ = set(list(range(self.L.shape[1])))
        self.P_ = set([145, 150, 152])

        self.U_ = set([])

    def RunCopyCatch(self):
        """Runs the main CopyCatch loop"""
#        self.ReadjustC()
        
        cnt = 1
        while True:
            P_l = self.P_
            cl = self.c
            self.c = self.UpdateCenter(self.c, self.P_)
            self.P_ = self.UpdateSubspace(self.c, self.P_)
            
            cnt = cnt + 1
            
            if cnt > 100:
                print("\nExiting due to non-convergence\n")
                break

            if ((self.c == cl).all()) and (P_l == self.P_):
                break
        
        return [self.c, self.P_, self.U_]

    def UpdateCenter(self, c, P_):
        """Updates cluster center"""
        U_ = self.FindUsers(self.U, c, P_)
        
        if len(U_) == 0: #if no user is found then return
            return c

        c_ = np.squeeze(np.asarray(np.sum(self.L[list(U_)], axis=0) / len(U_)))

        t = np.zeros((len(c_),), dtype=np.float64)
        for j in P_:
            U_, w = self.FindUsers(self.U, c, P_, j, 2.0*self.dt)
            U__, t[j] = self.FindCenter(U_, w, j)
            c_[j] = t[j]

        return c_

    def UpdateSubspace(self, c, P_l):
        """Updates subspace"""
        P_ = P_l
        U_ = self.FindUsers(self.U, c, P_l)
        for j_ in P_l:
            j__ = j_
            U_j__ = self.FindUsers(U_, c, set([j__]))

            for j in (self.P - P_):
                U_j = self.FindUsers(U_, c, set([j]))#(P_ - set([j__])) | set([j]) )
                if (U_j__ <= U_j) and (len(U_j) > 0):
                    j__ = j
                    U_j__ = U_j

            #P_ = (P_ - set([j_])) | set([j__])
            P_ = P_ | set([j__])

        self.U_ = U_j__
        return P_

    def FindCenter(self, U, w, jc):
        """Find weighted center of U in dimension jc """
        U_sorted = sorted ( U, key = lambda x:self.L[x, jc] )
        sum_w = 0
        U_ = []

        for i in U_sorted:
            t_max = self.L[i, jc]+(self.beta*self.dt)
            t_min = self.L[i, jc]
            U_temp = [x for x in U_sorted if t_min<=self.L[x, jc]<=t_max]

            sum_w_temp = sum([w[x] for x in U_temp])
            if sum_w < sum_w_temp:
                sum_w = sum_w_temp
                U_ = U_temp

        cj = (self.L[U_[0], jc] + self.L[U_[-1], jc]) / 2.0
        return [U_, cj]

    def FindUsers(self, U, c, P_, jc = -1, dt_ = -1.0):
        """Find users from U based on c and P_"""
        U_ = set([])
        w = np.zeros((len(self.U),), dtype=np.int)
        for i in U:
            for j in P_:
                if (self.I[i, j] == 1) and\
                    ( (np.abs( c[j] - self.L[i,j] ) <= self.dt) or\
                    ( (j==jc) and np.abs( c[j] - self.L[i,j] ) <= dt_ ) ):
                    w[i] = w[i] + 1

            if w[i] >= (self.phi * self.m) or (len(P_) <= 1 and w[i] > 0):
                U_ = U_ | set([i])

        if jc < 0 and dt_ < 0: #in case no jc and dt_ were given
            return U_

        return [U_, w]
    
    def ReadjustC(self):
        self.c = np.zeros(self.L.shape[1], dtype=np.float64)
        min_time = np.min( [x for x in self.L.flatten() if x > 0.0] )
        max_time = np.max( [x for x in self.L.flatten() if x > 0.0] )
#        P_list = list(self.P_)
#        np.random.shuffle(P_list)
#        P_list = P_list[:self.m]
        for P_ind in range(2):#P_list:
            self.c[P_ind] = min_time + (np.random.rand() * (max_time - min_time) )
        
        print ([x for x in self.c if x > 0.0])
    
    def readmatrixfile(self, path, make_model = False):
        """Reads and optionally saves the binarized version of the matrix"""
        f = open(path, 'r')
        x = f.readline()
        c = x.count(',') + 1
        f.close()
        
        mat = np.zeros((0, c), dtype = np.float64)
        
        f = open(path, 'r')
        for x in f:
            p = ast.literal_eval(x)
            mat = np.append(mat, [p], axis = 0)
        
        f.close()
        
        if make_model:
            output = open(path + '.pkl', 'wb')
            pickle.dump(mat, output)
            output.close()
        
        return mat
    
    def readserializedmatrix(self, path):
        """Reads binarized version of the matrix"""
        pkl_file = open(path, 'rb')
        return pickle.load(pkl_file)