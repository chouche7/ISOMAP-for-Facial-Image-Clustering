#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:48:39 2022

@author: chouche7
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path
import scipy.sparse.linalg as ll
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import math
#from sklearn.metrics import pairwise_distances
    
images = scipy.io.loadmat('data/faces.mat')['images'].T

##########################PART A##############################
#Step 1: Build a weighted graph A using nearest neighbors based on epsilon 
A = np.zeros((images.shape[0], images.shape[0]))
    
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        d = np.sqrt(sum((images[i] - images[j])**2))
        A[i][j] = d
        A[j][i] = d

def iso_map(A, epsilon):
    A[np.where(A > epsilon)] = 0

    #Step 2: Compute pairwise shortest distance matrix D
    D = shortest_path(A, directed=False)
    
    # visualize the weighted adjacency matrix
    plt.imshow(A)
    plt.title('Nearest Neighbor Graph')
    plt.show()
    
    # visualize the shortest distance matrix
    plt.imshow(D)
    plt.title('Shortest Distance Matrix')
    plt.show()
    
    # plot 4 random images
    images_id = np.random.randint(0, len(images), 4)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(images[images_id[0]].reshape(64,64).T, cmap = 'gray')
    axs[0, 0].set_title('Image' + str(images_id[0]))
    axs[0, 0].axis('off')
    axs[0, 1].imshow(images[images_id[1]].reshape(64,64).T, cmap = 'gray')
    axs[0, 1].set_title('Image' + str(images_id[1]))
    axs[0, 1].axis('off')
    axs[1, 0].imshow(images[images_id[2]].reshape(64,64).T, cmap = 'gray')
    axs[1, 0].set_title('Image' + str(images_id[2]))
    axs[1, 0].axis('off')
    axs[1, 1].imshow(images[images_id[3]].reshape(64,64).T, cmap = 'gray')
    axs[1, 1].set_title('Image' + str(images_id[3]))
    axs[1, 1].axis('off')
    
    ##########################PART B##############################
    #Step 3: Use a centering matrix H = I - 1/m11^T to get
    #C = -0.5 H (D)^@ H
    H = np.identity(D.shape[0]) - 1/(D.shape[0]) * np.ones_like(D)
    C = -0.5 * H @ (D ** 2) @ H
    
    #Step 4: Compute leading eigenvectors W and eigenvalues LAMBDA of C
    lam,W = ll.eigs(C,k = 2)
    lam = lam.real
    W = W.real
    
    # calculate the new coordinates Z
    Z = W @ np.diag(np.sqrt(lam))
    
    # select random images from dataset
    images_id = np.random.randint(0, len(images), 30)
        
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(Z[:, 0], Z[:, 1])
    ax.set_title('2D Components from Isomap of Facial Images')
    ax.set_ylabel('Down to Up')
    ax.set_xlabel('Left to Right')
    
    for i in images_id:
        face_image = images[i].reshape(64, 64).T
        image = OffsetImage(face_image, zoom=0.3, cmap = 'gray')
        annot = AnnotationBbox(image, Z[i], pad = 0.01, xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
        ax.add_artist(annot)

    plt.show()
    
np.random.seed(1)
#iso_map(A, 20)
#iso_map(A, 18)
#iso_map(A, 16)
#iso_map(A, 14)
iso_map(A, 12)

####################PART C######################
#Step 1: calculate sample mean and covariance matrix
mu = np.mean(images, axis = 0)
xc = (images - mu).T
C = np.dot(xc, xc.T)/(xc.shape[1])

#Step 2: find the eigenvectors corresponding to the largest eigenvalues
lam,W = ll.eigs(C,k = 2)
lam = lam.real
W = W.real

#first principal component
dim1 = np.dot(W[:,0].T,xc)/math.sqrt(lam[0])

#second principal component
dim2 = np.dot(W[:,1].T,xc)/math.sqrt(lam[1]) # extract 2nd eigenvalue

plt.figure(figsize=(8,8))
plt.scatter(dim1,dim2, marker = "o", c = "red")
plt.title('2D Components from Isomap of Facial Images')

# select random images from dataset
np.random.seed(1)
images_id = np.random.randint(0, len(images), 30)
        
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(dim1, dim2)
ax.set_title('2D Components from PCA of Facial Images')
ax.set_ylabel('Component 2') 
ax.set_xlabel('Component 1') 
    
for i in images_id:
   face_image = images[i].reshape(64, 64).T
   image = OffsetImage(face_image, zoom=0.3, cmap = 'gray')
   annot = AnnotationBbox(image, np.stack((-dim1,dim2)).T[i], pad = 0.01, xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
   ax.add_artist(annot)

plt.show()














