#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 

x = np.arange(50)[:, np.newaxis]
y = np.array([i + np.random.rand() for i in range(50)])[:, np.newaxis]
d = 2

def design_matrix(x, y, d):

    """
    Function for setting up a design X-matrix with rows [1, x, y, x², y², xy, ...]
    Input: x and y mesh, keyword argument k/n/d is the degree.
    """
    
    if len(x.shape) > 1:
    # reshape input to 1D arrays (easier to work with)
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((d+1)*(d+2)/2)	# number of elements in beta
    X = np.ones((N,p))

    for n in range(1,d+1):
        q = int((n)*(n+1)/2)
        for m in range(n+1):
            X[:,q+m] = x**(n-m)*y**m

    return X


