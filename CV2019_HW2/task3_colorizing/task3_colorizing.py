#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob


# Split the RGB directly

# In[31]:


icon = cv2.imread('icon.tif')
icon.shape
#print(icon[0])
print(icon.shape)
c1 = icon[:3244,:,:]
c2 = icon[3244:6488,:,:]
c3 = icon[6488: ,:,:]
plt.imshow(c1)
#plt.imshow(icon)


# In[32]:


plt.imshow(icon)


# Add RGB directly

# In[33]:


result = c1 + c2 + c3
print(c1.shape)
print(c2.shape)
print(c3.shape)
plt.imshow(result)


# Use NCC to align RGB

# In[6]:


from PIL import Image, ImageChops
from scipy.misc import imresize


# In[28]:


imname='icon.tif'
img=Image.open(imname)
img=np.asarray(img)
w,h=img.shape
print(w,h)

image_stack=[]

for i in range(1,5):
    print(i)
    scale=1/(i**2)
    image=imresize(img,scale)
    image_stack.append(image)
image_stack.reverse()
for im in image_stack:
    print (im.shape)
    
def ncc(a,b):
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))

def nccAlign(a, b, x,y,t):
    min_ncc = -1
    ivalue=np.linspace(-t+x,t+x,2*t,dtype=int)
    jvalue=np.linspace(-t+y,t+y,2*t,dtype=int)
    for i in ivalue:
        print(i)
        for j in jvalue:
            print(j)
            nccDiff = ncc(a,np.roll(b,[i,j],axis=(0,1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i,j]
    print(output)
    return output

for i in range(4):
    print(i)
    img=image_stack[i]
    print(img.shape)
    w,h=img.shape
    height=int(w/3)
    blue=img[0:height,:]
    green=img[height:2*height,:]
    red=img[2*height:3*height,:]
    x_gtob,y_gtob=0,0
    x_rtob,y_rtob=0,0
    alignGtoB = nccAlign(blue,green,x_gtob,y_gtob,5)
    alignRtoB = nccAlign(blue,red,x_rtob,y_rtob,5)
    x_gtob,y_gtob=alignGtoB[0]*2,alignGtoB[1]*2
    x_rtob,y_rtob=alignRtoB[0]*2,alignRtoB[1]*2

g=np.roll(green,[x_gtob,y_gtob],axis=(0,1))
r=np.roll(red,[x_rtob,y_rtob],axis=(0,1))
coloured = (np.dstack((r,g,blue))).astype(np.uint8)
coloured=coloured[int(coloured.shape[0]*0.05):int(coloured.shape[0]-coloured.shape[0]*0.05),int(coloured.shape[1]*0.05):int(coloured.shape[1]-coloured.shape[1]*0.05)]
coloured = Image.fromarray(coloured)
coloured.save('icon_aligned.jpg')
#plt.figure()
plt.imshow(coloured)


# In[24]:


imname='tobolsk.jpg'
img=Image.open(imname)
img=np.asarray(img)
w,h=img.shape
print(w,h)
image_stack=[]
for i in range(1,5):
    print(i)
    scale=1/(i**2)
    image=imresize(img,scale)
    image_stack.append(image)
image_stack.reverse()
for im in image_stack:
    print (im.shape)
    
def ncc(a,b):
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))

def nccAlign(a, b, x,y,t):
    min_ncc = -1
    ivalue=np.linspace(-t+x,t+x,2*t,dtype=int)
    jvalue=np.linspace(-t+y,t+y,2*t,dtype=int)
    for i in ivalue:
        print(i)
        for j in jvalue:
            print(j)
            nccDiff = ncc(a,np.roll(b,[i,j],axis=(0,1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i,j]
    print(output)
    return output

for i in range(4):
    print(i)
    img=image_stack[i]
    print(img.shape)
    w,h=img.shape
    height=int(w/3)
    blue=img[:height,:]
    green=img[height:2*height,:]
    red=img[2*height:3*height,:]
    x_gtob,y_gtob=0,0
    x_rtob,y_rtob=0,0
    alignGtoB = nccAlign(blue,green,x_gtob,y_gtob,5)
    alignRtoB = nccAlign(blue,red,x_rtob,y_rtob,5)
    x_gtob,y_gtob=alignGtoB[0]*2,alignGtoB[1]*2
    x_rtob,y_rtob=alignRtoB[0]*2,alignRtoB[1]*2

g=np.roll(green,[x_gtob,y_gtob],axis=(0,1))
r=np.roll(red,[x_rtob,y_rtob],axis=(0,1))
coloured = (np.dstack((r, g, blue))).astype(np.uint8)
coloured=coloured[int(coloured.shape[0]*0.05):int(coloured.shape[0]-coloured.shape[0]*0.1),int(coloured.shape[1]*0.05):int(coloured.shape[1]-coloured.shape[1]*0.1)]
coloured = Image.fromarray(coloured)
coloured.save('tobolsk_aligned.jpg')
#plt.figure()
plt.imshow(coloured)


# In[25]:


imname='monastery.jpg'
img=Image.open(imname)
img=np.asarray(img)
w,h=img.shape
print(w,h)
image_stack=[]
for i in range(1,5):
    print(i)
    scale=1/(i**2)
    image=imresize(img,scale)
    image_stack.append(image)
image_stack.reverse()
for im in image_stack:
    print (im.shape)
    
def ncc(a,b):
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))

def nccAlign(a, b, x,y,t):
    min_ncc = -1
    ivalue=np.linspace(-t+x,t+x,2*t,dtype=int)
    jvalue=np.linspace(-t+y,t+y,2*t,dtype=int)
    for i in ivalue:
        print(i)
        for j in jvalue:
            print(j)
            nccDiff = ncc(a,np.roll(b,[i,j],axis=(0,1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i,j]
    print(output)
    return output

for i in range(4):
    print(i)
    img=image_stack[i]
    print(img.shape)
    w,h=img.shape
    height=int(w/3)
    blue=img[:height,:]
    green=img[height:2*height,:]
    red=img[2*height:3*height,:]
    x_gtob,y_gtob=0,0
    x_rtob,y_rtob=0,0
    alignGtoB = nccAlign(blue,green,x_gtob,y_gtob,5)
    alignRtoB = nccAlign(blue,red,x_rtob,y_rtob,5)
    x_gtob,y_gtob=alignGtoB[0]*2,alignGtoB[1]*2
    x_rtob,y_rtob=alignRtoB[0]*2,alignRtoB[1]*2

g=np.roll(green,[x_gtob,y_gtob],axis=(0,1))
r=np.roll(red,[x_rtob,y_rtob],axis=(0,1))
coloured = (np.dstack((r, g, blue))).astype(np.uint8)
coloured=coloured[int(coloured.shape[0]*0.05):int(coloured.shape[0]-coloured.shape[0]*0.1),int(coloured.shape[1]*0.05):int(coloured.shape[1]-coloured.shape[1]*0.1)]
coloured = Image.fromarray(coloured)
coloured.save('monastery_aligned.jpg')
#plt.figure()
plt.imshow(coloured)


# In[26]:


imname='nativity.jpg'
img=Image.open(imname)
img=np.asarray(img)
w,h=img.shape
print(w,h)
image_stack=[]
for i in range(1,5):
    print(i)
    scale=1/(i**2)
    image=imresize(img,scale)
    image_stack.append(image)
image_stack.reverse()
for im in image_stack:
    print (im.shape)
    
def ncc(a,b):
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))

def nccAlign(a, b, x,y,t):
    min_ncc = -1
    ivalue=np.linspace(-t+x,t+x,2*t,dtype=int)
    jvalue=np.linspace(-t+y,t+y,2*t,dtype=int)
    for i in ivalue:
        print(i)
        for j in jvalue:
            print(j)
            nccDiff = ncc(a,np.roll(b,[i,j],axis=(0,1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i,j]
    print(output)
    return output

for i in range(4):
    print(i)
    img=image_stack[i]
    print(img.shape)
    w,h=img.shape
    height=int(w/3)
    blue=img[:height,:]
    green=img[height:2*height,:]
    red=img[2*height:3*height,:]
    x_gtob,y_gtob=0,0
    x_rtob,y_rtob=0,0
    alignGtoB = nccAlign(blue,green,x_gtob,y_gtob,5)
    alignRtoB = nccAlign(blue,red,x_rtob,y_rtob,5)
    x_gtob,y_gtob=alignGtoB[0]*2,alignGtoB[1]*2
    x_rtob,y_rtob=alignRtoB[0]*2,alignRtoB[1]*2

g=np.roll(green,[x_gtob,y_gtob],axis=(0,1))
r=np.roll(red,[x_rtob,y_rtob],axis=(0,1))
coloured = (np.dstack((r, g, blue))).astype(np.uint8)
coloured=coloured[int(coloured.shape[0]*0.05):int(coloured.shape[0]-coloured.shape[0]*0.1),int(coloured.shape[1]*0.05):int(coloured.shape[1]-coloured.shape[1]*0.1)]
coloured = Image.fromarray(coloured)
coloured.save('nativity_aligned.jpg')
#plt.figure()
plt.imshow(coloured)


# In[27]:


imname='cathedral.jpg'
img=Image.open(imname)
img=np.asarray(img)
w,h=img.shape
print(w,h)
image_stack=[]
for i in range(1,5):
    print(i)
    scale=1/(i**2)
    image=imresize(img,scale)
    image_stack.append(image)
image_stack.reverse()
for im in image_stack:
    print (im.shape)
    
def ncc(a,b):
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))

def nccAlign(a, b, x,y,t):
    min_ncc = -1
    ivalue=np.linspace(-t+x,t+x,2*t,dtype=int)
    jvalue=np.linspace(-t+y,t+y,2*t,dtype=int)
    for i in ivalue:
        print(i)
        for j in jvalue:
            print(j)
            nccDiff = ncc(a,np.roll(b,[i,j],axis=(0,1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i,j]
    print(output)
    return output

for i in range(4):
    print(i)
    img=image_stack[i]
    print(img.shape)
    w,h=img.shape
    height=int(w/3)
    blue=img[:height,:]
    green=img[height:2*height,:]
    red=img[2*height:3*height,:]
    x_gtob,y_gtob=0,0
    x_rtob,y_rtob=0,0
    alignGtoB = nccAlign(blue,green,x_gtob,y_gtob,5)
    alignRtoB = nccAlign(blue,red,x_rtob,y_rtob,5)
    x_gtob,y_gtob=alignGtoB[0]*2,alignGtoB[1]*2
    x_rtob,y_rtob=alignRtoB[0]*2,alignRtoB[1]*2

g=np.roll(green,[x_gtob,y_gtob],axis=(0,1))
r=np.roll(red,[x_rtob,y_rtob],axis=(0,1))
coloured = (np.dstack((r, g, blue))).astype(np.uint8)
coloured=coloured[int(coloured.shape[0]*0.05):int(coloured.shape[0]-coloured.shape[0]*0.1),int(coloured.shape[1]*0.05):int(coloured.shape[1]-coloured.shape[1]*0.1)]
coloured = Image.fromarray(coloured)
coloured.save('cathedral_aligned.jpg')
#plt.figure()
plt.imshow(coloured)

