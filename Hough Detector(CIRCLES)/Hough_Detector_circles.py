
import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow
import matplotlib.pylab as plt
import math as math
from collections import defaultdict
from itertools import product



def dnorm(x, mu, sd): # Completementary function of the Gaussian Kernel
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1): # Construction of the Gaussian Kernel
 
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D


def convolution(image,filter) : # Convolution Function with no Edge Handling (zeroing)
  n=filter.shape[0]
  n=int(np.floor(n/2))
  newimage=np.zeros(image.shape)
  #for x in range(0,image.shape[2]):
  for i in range(n,image.shape[0]-n):
    for j in range(n,image.shape[1]-n):
      newimage[i,j]=np.sum(image[i-n:i+n+1,j-n:j+n+1]*filter)
  newimage *= 255.0 / newimage.max()
  return newimage

def Sobel(img): # Filtre de Sobel
  filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  cx=convolution(img,filter)
  cy=convolution(img, np.flip(filter.T, axis=0))
  gradient_magnitude = np.sqrt(np.square(cx) + np.square(cy))
  gradient_magnitude *= 255.0 / gradient_magnitude.max()
  return cx,cy,gradient_magnitude

def contour_Pix(gradient_magnitude,t): # Edge Detection function using Sobel

  contours=[]
  for i in range (1,gradient_magnitude.shape[0]-1):
    for j in range(1,gradient_magnitude.shape[1]-1):
      if (gradient_magnitude[i,j]>t*255) :
        contours.append([i,j])
  contours=np.array(contours)
  return contours
# and gradient_magnitude[i,j,1]>t*255 and gradient_magnitude[i,j,2]>t*255
def init_Acc(image): # J'ai pas utilisé cette fonction car j'ai utilisé une dictionnaire.
  acc=[[[0 for x in range(5,int(np.sqrt(pow(image.shape[0],2)+pow(image.shape[1],2)))+1)]for k in range(image.shape[1])]for i in range(image.shape[0])]
  acc=np.array(acc)
  return acc


def del_Sim(calc,maxes): # Function used to avoid taking centers that are contained within another circle.
  
  for a,b in calc:
    x,y,r=a
    if (all( (x-xc)**2 + (y-yc)**2 > (rc-2) ** 2 for xc,yc,rc in maxes )):
      return (x,y,r)

# Declaration of the different pictures and kernels.

img=cv.imread('/content/drive/MyDrive/TP1_IMAGE_PROCESS/four.png')
image1_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY )
kernel_size=5
kernel=gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
gradient_magnitude=convolution(image1_gray,kernel)
cx,cy,gradient_magnitude=Sobel(image1_gray)
cv2_imshow(gradient_magnitude)
contours=contour_Pix(gradient_magnitude,0.62)


acc=defaultdict(int)  # Initiating a dictionary with 0 values as default.


gradient_Contours=[]



def uniq(lst): # Delete duplicated values in a list.
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l): # Pareil 
    return list(uniq(sorted(l, reverse=True)))

for a,b in contours: # Speed up function, it basically reduces the number of edge points that are used in the next calculations, using the gradient vector's direction.
  i=0
  while (i<22): # 22 is a coefficient that I picked considering the circles' radius.
    gradient_Contours.append([a+int(i*cy[a, b]/gradient_magnitude[a, b]),b-int(i*cx[a, b]/gradient_magnitude[a, b])])
    i=i+2.2 # Paramétre qu'on peut changer


gradient_Contours=sort_and_deduplicate(gradient_Contours) # Delete duplicated items in the list.


for x,y in contours:
    for j,k in gradient_Contours:
        if (int((x-j)**2+(y-k)**2)>9):
          acc[j,k,int(np.sqrt((x-j)**2+(y-k)**2))]+=1/(int(np.sqrt((x-j)**2+(y-k)**2))) # Normalisation par division sur le rayon



sorted_acc=(sorted(acc.items(), key=lambda i: -i[1])) # On trie l'accumulateur

maxes=[]

for i in range (0,5): # On prend 4 cercles de l'accumulateur.
  b_key = del_Sim(sorted_acc,maxes)
  maxes.append(b_key)

maxes=np.array(maxes)

lmao=cv.imread('/content/drive/MyDrive/TP1_IMAGE_PROCESS/four.png') # On trace les cercles.

for j in range (0,4):

  cv.circle(lmao,(maxes[j,1], maxes[j,0]), maxes[j,2], (0,0,255))
plt.figure()
plt.imshow(lmao)

sorted_acc
