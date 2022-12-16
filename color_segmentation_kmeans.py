#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 21:33:21 2022

@author: chouche7
"""

from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

#Q2 kmeans algorithm utilizing l2 norm
def kmeans_l2(pixels, k):
    
    start_time = time.time()
    
    #flatten the pixels
    v_img = pixels.reshape(-1, 3)
    
    #assign random pixel as centroid using broadcasting
    c = v_img[np.random.randint(0, len(v_img), k),:]
    c = np.reshape(c,[k,1,3])
    x = np.reshape(v_img, [1,len(v_img),3])
    
    #keep track of the iterations
    count = 1
    prev_c = c
    
    while True:    
        
        #calculate distance change
        d = np.sqrt(np.sum(np.square(x - c), axis = 2))
        
        #create one hot encoding of assigned classes for better calculation of new centroid
        pi = np.argmin(d, axis = 0)
        one_hot_pi = np.zeros((pi.size, pi.max()+1))
        one_hot_pi[np.arange(pi.size), pi] = 1
        pi = np.reshape(one_hot_pi.T,[k,1,len(v_img)])
        
        #calculate new center based on mean of classes
        c = np.around((pi @ x)/np.reshape(np.sum(one_hot_pi, axis = 0), [k, 1, 1]),0)
        
        #if cluster center no longer changes
        if np.array_equal(prev_c, c):
            break
        
        prev_c = c
        count+= 1
        
    pi = np.argmin(d, axis = 0)
    c = np.reshape(c, [k, 3])
    pix_compress = c[pi].reshape(pixels.shape)
    
    return pi, c, count, (time.time() - start_time), pix_compress

#football = np.array(Image.open('data/football.bmp'))
hestain = np.array(Image.open('data/hestain.bmp'))
tumor = np.array(Image.open('data/tumor.bmp'))

#set seed
np.random.seed(123)

#hestain euclidean distance
pi1, centers1, count1, runtime1, img1 = kmeans_l2(hestain, 2)    
pi2, centers2, count2, runtime2, img2 = kmeans_l2(hestain, 4)        
pi3, centers3, count3, runtime3, img3 = kmeans_l2(hestain, 6) 

f, axarr = plt.subplots(2,2)
plt.suptitle("Hestain (l2-norm)")
axarr[0,0].imshow(hestain)
axarr[0,0].axis('off')
axarr[0,0].title.set_text('original image')
axarr[0,1].imshow(img1.astype(np.uint8))
axarr[0,1].axis('off')
axarr[0,1].title.set_text('compressed k = 2')
axarr[1,0].imshow(img2.astype(np.uint8))
axarr[1,0].axis('off')
axarr[1,0].title.set_text('compressed k = 4')
axarr[1,1].imshow(img3.astype(np.uint8))
axarr[1,1].axis('off')
axarr[1,1].title.set_text('compressed k = 6')



# #football euclidean distance
# pi1, centers1, count1, runtime1, img1 = kmeans_l2(football, 2)    
# pi2, centers2, count2, runtime2, img2 = kmeans_l2(football, 4)        
# pi3, centers3, count3, runtime3, img3 = kmeans_l2(football, 8) 

# f, axarr = plt.subplots(2,2)
# plt.suptitle("Football (l2-norm)")
# axarr[0,0].imshow(football)
# axarr[0,0].axis('off')
# axarr[0,0].title.set_text('original image')
# axarr[0,1].imshow(img1.astype(np.uint8))
# axarr[0,1].axis('off')
# axarr[0,1].title.set_text('compressed k = 2')
# axarr[1,0].imshow(img2.astype(np.uint8))
# axarr[1,0].axis('off')
# axarr[1,0].title.set_text('compressed k = 4')
# axarr[1,1].imshow(img3.astype(np.uint8))
# axarr[1,1].axis('off')
# axarr[1,1].title.set_text('compressed k = 8')


#tumor euclidean distance (self-selected)
pi1, centers1, count1, runtime1, img1 = kmeans_l2(tumor, 2)    
pi2, centers2, count2, runtime2, img2 = kmeans_l2(tumor, 4)        
pi3, centers3, count3, runtime3, img3 = kmeans_l2(tumor, 6) 

f, axarr = plt.subplots(2,2)
plt.suptitle("Tumor (l2-norm)")
axarr[0,0].imshow(tumor)
axarr[0,0].axis('off')
axarr[0,0].title.set_text('original image')
axarr[0,1].imshow(img1.astype(np.uint8))
axarr[0,1].axis('off')
axarr[0,1].title.set_text('compressed k = 2')
axarr[1,0].imshow(img2.astype(np.uint8))
axarr[1,0].axis('off')
axarr[1,0].title.set_text('compressed k = 4')
axarr[1,1].imshow(img3.astype(np.uint8))
axarr[1,1].axis('off')
axarr[1,1].title.set_text('compressed k = 6')



# #kmeans algorithm utilizing Manhattan norm
# def kmeans_manhattan(pixels, k):
    
#     start_time = time.time()
    
#     #flatten the pixels
#     v_img = pixels.reshape(-1, 3)
    
#     #assign random pixel as centroid using broadcasting
#     c = v_img[np.random.randint(0, len(v_img), k),:]
#     c = np.reshape(c,[k,1,3])
#     x = np.reshape(v_img, [1,len(v_img),3])
    
#     #keep track of the iterations
#     count = 1
#     prev_c = c

#     while True:    
        
#         #calculate distance change
#         d = np.sum(np.absolute(x - c), axis = 2)
        
#         #create one hot encoding of assigned classes for better calculation of new centroid
#         pi = np.argmin(d, axis = 0)

#         #calculate new center based on the median
#         new_c = np.zeros((k, 3))
        
#         for i in range(k):
            
#             #find the minimum distance to a cluster for each pixel
#             min_d = np.min(d, axis =  0)
            
#             #find the median distance for each cluster center
#             med_d = np.median(min_d[np.where(pi == i)],axis = 0)
            
#             #pick one index that is the median for each cluster
#             idx = np.array(np.where(min_d == med_d)[0])
            
#             #if there is nothing in the index then we terminate the algorithm
#             if idx.size == 0:
#                 break
            
#             #randomly select one of the indexes if there are multiple
#             idx = np.random.choice(idx)

#             #assign new center based on the median index
#             new_c[i]= v_img[int(idx),:]
            
#         #if there is no index median, then we have reached the end 
#         if idx.size == 0:
#             break
        
#         c = np.reshape(new_c,[k,1,3])
        
#         #if cluster center no longer changes
#         if np.array_equal(prev_c, c):
#             break

#         prev_c = c
#         count+= 1
        
#     pi = np.argmin(d, axis = 0)
#     c = np.reshape(c, [k, 3])
#     pix_compress = c[pi].reshape(pixels.shape)
    
#     return pi, c, count, (time.time() - start_time), pix_compress


# #set seed
# np.random.seed(8)

# #hestain manhattan distance
# pi1, centers1, count1, runtime1, img1 = kmeans_manhattan(hestain, 2)    
# pi2, centers2, count2, runtime2, img2 = kmeans_manhattan(hestain, 4)        
# pi3, centers3, count3, runtime3, img3 = kmeans_manhattan(hestain, 6) 

# f, axarr = plt.subplots(2,2)
# plt.suptitle("Hestain (Manhattan)")
# axarr[0,0].imshow(hestain)
# axarr[0,0].axis('off')
# axarr[0,0].title.set_text('original image')
# axarr[0,1].imshow(img1.astype(np.uint8))
# axarr[0,1].axis('off')
# axarr[0,1].title.set_text('compressed k = 2')
# axarr[1,0].imshow(img2.astype(np.uint8))
# axarr[1,0].axis('off')
# axarr[1,0].title.set_text('compressed k = 4')
# axarr[1,1].imshow(img3.astype(np.uint8))
# axarr[1,1].axis('off')
# axarr[1,1].title.set_text('compressed k = 8')

# #set seed
# np.random.seed(1)

# # #football manhattan distance
# # pi1, centers1, count1, runtime1, img1 = kmeans_manhattan(football, 2)    
# # pi2, centers2, count2, runtime2, img2 = kmeans_manhattan(football, 4)        
# # pi3, centers3, count3, runtime3, img3 = kmeans_manhattan(football, 8) 

# # f, axarr = plt.subplots(2,2)
# # plt.suptitle("Football (Manhattan)")
# # axarr[0,0].imshow(football)
# # axarr[0,0].axis('off')
# # axarr[0,0].title.set_text('original image')
# # axarr[0,1].imshow(img1.astype(np.uint8))
# # axarr[0,1].axis('off')
# # axarr[0,1].title.set_text('compressed k = 2')
# # axarr[1,0].imshow(img2.astype(np.uint8))
# # axarr[1,0].axis('off')
# # axarr[1,0].title.set_text('compressed k = 4')
# # axarr[1,1].imshow(img3.astype(np.uint8))
# # axarr[1,1].axis('off')
# # axarr[1,1].title.set_text('compressed k = 8')

# #set seed
# np.random.seed(4)

# #tumor manhattan distance
# pi1, centers1, count1, runtime1, img1 = kmeans_manhattan(tumor, 2)    
# pi2, centers2, count2, runtime2, img2 = kmeans_manhattan(tumor, 4)        
# pi3, centers3, count3, runtime3, img3 = kmeans_manhattan(tumor, 6) 

# f, axarr = plt.subplots(2,2)
# plt.suptitle("Tumor (Manhattan)")
# axarr[0,0].imshow(tumor)
# axarr[0,0].axis('off')
# axarr[0,0].title.set_text('original image')
# axarr[0,1].imshow(img1.astype(np.uint8))
# axarr[0,1].axis('off')
# axarr[0,1].title.set_text('compressed k = 2')
# axarr[1,0].imshow(img2.astype(np.uint8))
# axarr[1,0].axis('off')
# axarr[1,0].title.set_text('compressed k = 4')
# axarr[1,1].imshow(img3.astype(np.uint8))
# axarr[1,1].axis('off')
# axarr[1,1].title.set_text('compressed k = 8')
