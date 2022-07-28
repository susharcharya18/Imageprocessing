# Imageprocessing
http://localhost:8891/notebooks/New%20folder/programp1.ipynb<br>
1)Develope a program to display greyscale image using read and write operation<br>
import cv2<br>
img=cv2.imread('flw1.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/173812703-83a9050e-d7ab-4586-b13b-d21bf6856040.png)<br>
2) Develope a program to display the image using matplotlib<br> 
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('leaf1.jpg')<br>
plt.imshow(img)<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/173813368-136d4654-07a2-40c2-8f6f-d616996be0e9.png)<br>
3)Develop a program to perform Linear Transformation<br>
a) Rotation b) Scaling<br>
from PIL import Image<br>
img=Image.open('bf1.jpg')<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/173813810-4609a005-d140-43af-a3a6-defd18a78376.png)<br>
4)Develop a program to convert Colorstring to RGB color value<br>
from PIL import ImageColor<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/173814159-22c9431c-03ac-485b-b6f7-214a880d96d2.png)<br>
5)Write a program to create image using colors<br>
from PIL import Image<br>
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/173814503-e00acde1-af74-4c26-89f6-bae047e2f63e.png)<br>
6)Develop a program to visualize the image using various Colorspaces<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('bf3.jpg')<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/173814697-38f7d778-75bd-4973-baff-14cdfbc69aea.png)<br>
7)Write a program to display the image attributes<br>
from PIL import Image<br>
image=Image.open('flw3.jpg')<br>
print("Filename",image.filename)<br>
print("Format",image.format)<br>
print("Mode",image.mode)<br>
print("Size",image.size)<br>
print("Width",image.width)<br>
print("Height",image.height)<br>
image.close();<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/173815295-6f31699b-4609-44e8-9210-3bb4319a013e.png)<br>
8)Convert the original image to gray scale and then to binary<br>
import cv2<br>
#read the image file<br>
img=cv2.imread("flw22.jpg")<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
#gray scale<br>
img=cv2.imread("flw22.jpg",0)<br>
cv2.imshow("gray",img)<br>
cv2.waitKey(0)<br>
#binary image<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("BINARY",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/174044748-26dfcc90-888b-4c63-abc3-853fd35d4ce4.png)<br>
![image](https://user-images.githubusercontent.com/97939356/174045047-19658c38-e426-4b83-9d5a-1d2c2ef158c2.png)<br>
![image](https://user-images.githubusercontent.com/97939356/174045302-fde2d346-0049-49a8-8da3-7c63501b68c8.png)<br>
9)Resize the original image<br>
import cv2<br>
img=cv2.imread('flw22.jpg')<br>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
cv2.waitKey(0)<br>
#to show the resized image<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/174043706-0a86ce89-0f01-4dcd-a402-43c03448d638.png)<br>
10)develop a program to readimage using URL<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://images.hindustantimes.com/rf/image_size_630x354/HT/p2/2019/08/08/Pictures/_6bda0940-b9ad-11e9-98cb-e738ad509720.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/175019563-1ef4c35d-eefc-4fcd-b4d8-536d10c3351b.png)<br>
11)Write a program to perform arithematic operation on images
import cv2
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
#reading image files<br>
img1=cv2.imread('bf1.jpg')<br>
img2=cv2.imread('bf3.jpg')<br>
#applying numpy addition on images<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
#saving the outout image<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
#saving the outout image<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
#saving the outout image<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
#saving the outout image<br>
cv2.imwrite('output.jpg',fimg4)<br>
output:<br>
1)![image](https://user-images.githubusercontent.com/97939356/175019733-b02ac822-3e74-4e0d-bf6c-cfe6fe70d17b.png)<br>
2)![image](https://user-images.githubusercontent.com/97939356/175019810-a4738a4a-b4e4-4310-b99f-3c67090277b9.png)<br>
3)![image](https://user-images.githubusercontent.com/97939356/175019905-57359335-99a4-4031-9f61-dc171bfc86dc.png)<br>
4)![image](https://user-images.githubusercontent.com/97939356/175019986-237aa09f-a0dc-47cd-af68-94212b230495.png)<br>
12)Wite a program to mask and blur the image <br>
import cv2<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('bf55.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97939356/175020094-8dcc0acd-4d36-47f4-9ba7-6c7b75d40479.png)<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97939356/175020213-da03bde8-8c74-4ddb-976c-63f7ec657714.png)<br>
light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97939356/175020311-4e0bf448-7b05-4807-9f1c-ea003ed934f3.png)<br>
final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_mask)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97939356/175020431-3b05d13d-3fd0-4c05-b19d-74fe7b60f0c0.png)<br>
blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>
![image](https://user-images.githubusercontent.com/97939356/175020540-fe6a5323-a391-4dbe-b705-0c8e9a38f1a3.png)<br>
13)develop the program to change the image to different color space<br>
import cv2 <br>
img=cv2.imread("bf1.jpg")<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/175263981-f61e0c85-07d2-48e3-9ace-ba221ba6ff72.png)<br>
![image](https://user-images.githubusercontent.com/97939356/175264136-a71009b9-93cf-45bf-a177-5b1a469e8658.png)<br>
![image](https://user-images.githubusercontent.com/97939356/175264337-20846ea4-c8ea-4c9a-b5da-ce30f19ff8a0.png)<br>
![image](https://user-images.githubusercontent.com/97939356/175264441-34c44e75-96e6-4132-99e7-70cb8f6148c1.png)<br>
![image](https://user-images.githubusercontent.com/97939356/175265162-149969a5-fb00-4ab4-bbf9-df224b75cf99.png)<br>
14)program to create an image using 2D array<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('flw4.jpg')<br>
img.show()<br>
c.waitKey(0)<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/175266876-0b317fda-3933-4a08-8532-51ce9dac7c1c.png)<br>

import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('PL1.jpg',1)<br>
image2=cv2.imread('PL1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAND=cv2.bitwise_and(image1,image2)<br>
bitwiseOR=cv2.bitwise_or(image1,image2)<br>
bitwiseXOR=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAND)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOR)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXOR)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/178717045-bca14314-5112-4f3a-92fa-0e2061ff77ca.png)<br>

#importing libraries<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('flo3.jpg')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>
#Gaussian Blur<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
#Median BLUR<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>
#Bilateral Blur<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/178717202-0bac06a0-73ae-4374-895a-ebc317aa0b2f.png)<br>
![image](https://user-images.githubusercontent.com/97939356/178717276-eb1bee96-23db-4457-8ebd-46ea8cfc1360.png)<br>
![image](https://user-images.githubusercontent.com/97939356/178717375-af5cbaac-1736-4029-94ab-c963ae2fc247.png)<br>
![image](https://user-images.githubusercontent.com/97939356/178717436-2b4a70fd-bbf5-46cf-bdce-5e8f6a49eddf.png)<br>

from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('flo4.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
color=1.5<br>
image_Contrasted=enh_con.enhance(color)<br>
image_Contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
Sharpness=1.0<br>
image_Sharped=enh_con.enhance(color)<br>
image_Sharped.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/178719513-907f72c9-1ef2-4305-844f-fd09ed52072f.png)<br>
![image](https://user-images.githubusercontent.com/97939356/178719602-918ad3a4-f90e-4462-b465-510ad9ff0883.png)<br>
![image](https://user-images.githubusercontent.com/97939356/178719678-56c0cda4-2f96-4e8e-b620-737d62524d18.png)<br>
![image](https://user-images.githubusercontent.com/97939356/178719742-f652d2d8-b728-4be0-bbb4-8ad69682d645.png)<br>
![image](https://user-images.githubusercontent.com/97939356/178719794-a2af0071-79b1-43b5-af6b-baf6aac6a719.png)<br>

import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('bf3.jpg',0)<br>
azx=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/178719059-90fbfaed-87c0-40e0-a416-17ab931f4df4.png)<br>


import cv2<br>
OriginalImg=cv2.imread('flo1.jpg')<br>
GrayImg=cv2.imread('flo1.jpg',0)<br>
isSaved=cv2.imwrite('C:/i.jpg',GrayImg)<br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('The Image is successfully saved')<br>
    OUTPUT:<br>
    ![image](https://user-images.githubusercontent.com/97939356/178699172-4f8be175-c60b-42c3-913f-583ce7482dae.png)<br>
    ![image](https://user-images.githubusercontent.com/97939356/178699314-dfe58cc5-989f-41de-8e8e-fc7da3d1cd37.png)<br>
   
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('bf1.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/178703256-2f841cde-b8d8-43bd-946e-d81b8e4be626.png)<br>

import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('bf2.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/178704124-234cd9ce-e2bc-4719-aca9-63b4ab6b4fee.png)<br>

import numpy as np<br>
import skimage.color<br>
import skimage.io<br>
import matplotlib.pyplot as plt<br>
#matplotlib widget<br>

#read the image of a plant seedling as grayscale from the outset<br>
image = skimage.io.imread(fname="img3.jpg", as_gray=True)<br>
image1 = skimage.io.imread(fname="img3.jpg")<br>
#display the image<br>
fig, ax = plt.subplots()<br>
plt.imshow(image, cmap="gray")<br>
plt.show()<br>

#display the image<br>
fig, ax = plt.subplots()<br>
plt.imshow(image1, cmap="gray")<br>
plt.show()<br>

#create the histogram<br>
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))<br>

#configure and draw the histogram figure<br>
plt.figure()<br>
plt.title("Grayscale Histogram")<br>
plt.xlabel("grayscale value")<br>
plt.ylabel("pixel count")<br>
plt.xlim([0.0, 1.0]) # <- named arguments do not work here<br>

plt.plot(bin_edges[0:-1], histogram) # <- or here<br>
plt.show()<br>
output:<br>
![image](https://user-images.githubusercontent.com/97939356/178964931-4c5ea6de-259b-412d-b070-dd10538bf680.png)<br>
![image](https://user-images.githubusercontent.com/97939356/178965018-e77ead41-848a-466a-9968-f36ba941608c.png)<br>
![image](https://user-images.githubusercontent.com/97939356/178965080-0d66b97b-4761-4c01-828d-225455299e08.png)<br>

Program to perform basic image data analysis using intensity transformation(Image negative,Log transformation,Gamma correction)<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('L1.jfif')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/180174362-09b538ca-81e8-4a02-816c-2e4fa0fdb017.png)<br>

negative=255-pic #neg=(L-1)-img<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/180174513-66ad4db9-2a03-4e21-90c3-1cc41eeb6401.png)<br>

%matplotlib inline<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>

pic=imageio.imread('LF1.jpg')<br>
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>

max_=np.max(gray)<br>

def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/180174661-a1a0fa97-020a-4be6-8fd4-fcff37ce2a22.png)<br>

import imageio<br>
import matplotlib.pyplot as plt<br>
#gamma encoding<br>
pic=imageio.imread('LF1.jpg')<br>
gamma=2.2#gamma<1~dark;gamma>~bright<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/180201095-4664a414-15f9-4d36-8351-c70454b0ac38.png)<br>


Program to perform basic image manipulation (sharpness,flipping,cropping)<br>
#IMAGE SHARPEN<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
#load the image<br>
my_image=Image.open('LF1.jpg')<br>
#use sharpen function<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
#SAVE THE IMAGE<br>
sharp.save('F:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/180201011-dbd070b4-00d6-4711-9a04-aa3f8c9827ec.png)<br>


#IMAGE FLIP<br>
import matplotlib.pyplot as plt<br>
from PIL import Image<br>
#load the image<br>
img=Image.open('L1.jfif')<br>
plt.imshow(img)<br>
plt.show()<br>
#USE THE FLIP FUNCTION<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>
#save the image<br>
flip.save('F:/image_flip.jfif')<br>
plt.imshow(flip)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/180200913-14dca94a-c583-458f-afe3-439bf0a6a97a.png)<br>
![image](https://user-images.githubusercontent.com/97939356/180200955-79d3eb71-7f47-4824-b4c5-120313e4e3f5.png)<br>


#IMAGE CROPPING<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
im=Image.open('L1.jfif')<br>
width,height=im.size<br>
im1=im.crop((50,25,175,200))<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/180200847-9821aa22-7da4-47ef-ad2e-f4b9d2e90335.png)<br>

from PIL import Image, ImageStat<br>

im = Image.open('b4.jfif')<br>
stat = ImageStat.Stat(im)<br>
print(stat.stddev)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/181441681-ceaddad0-0521-424b-826c-4403f168980b.png)<br>
<br>

import cv2<br>
import numpy as np<br>
img=cv2.imread('b4.jfif')<br>
cv2.imshow('b4.jfif',img)<br>
cv2.waitKey(0)<br>
#max_channels=np.amax([np.amax(img[:,:,0]),np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>
#print(max_channels)
np.max(img)<br>
OUTPUT:<br>
0<br>





import cv2<br>
import numpy as np<br>
img=cv2.imread('fl1.jfif')<br>
cv2.imshow('fl1.jfif',img)<br>
cv2.waitKey(0)<br>
#min_channels=np.amin([np.amin(img[:,:,0]),np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>
#print(min_channels)<br>
np.average(img)<br>
OUTPUT:<br>
178.26236885349513<br>

import cv2<br>
import numpy as np<br>
img=cv2.imread('fl1.jfif')<br>
cv2.imshow('fl1.jfif',img)<br>
cv2.waitKey(0)<br>
np.std(img)<br>
OUTPUT:<br>
59.692818587963856<br>




# Python3 program for printing<br>
# the rectangular pattern<br>
 
# Function to print the pattern<br>
def printPattern(n):<br>
 
    arraySize = n * 2 - 1;<br>
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>
         
    # Fill the values<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            if(abs(i - (arraySize // 2))<br> >
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2));<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2));<br>
             
    # Print the array<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br>
 
# Driver Code<br>
n = 4;<br>
 
printPattern(n);<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/181435621-9bb5acb7-d966-4048-9814-af1fec4fbfa5.png)<br>


from PIL import Image<br>
import numpy as np<br>
w, h = 1000, 1000<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:256, 0:256] = [204, 0, 0]<br>
data[257:512,0:256] = [0, 255, 0]<br>
data[513:780, 0:256] = [0, 0, 255]<br>
data[781:1000, 0:256] = [0, 125, 255]<br>
data[0:256, 257:512] = [255, 212, 0]<br>
data[0:256, 513:780] = [0, 212, 56]<br>
data[0:256, 781:1000] = [245, 0, 56]<br>
data[257:512,257:512] = [24, 5, 255]<br>
data[257:512,513:780] = [240, 52, 255]<br>
data[257:512,781:1000] = [40, 252, 255]<br>
data[513:780,257:512] = [140, 52, 255]<br>
data[781:1000,257:512] = [240, 152, 255]<br>
data[781:1000,513:780] = [40, 152, 255]<br>
data[781:1000,780:1000] = [240, 152, 255]<br>
data[513:780,513:780] = [200, 52, 55]<br>
data[513:780,781:1000] = [0, 252, 155]<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('b4.jfif')<br>
img.show()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/181438687-5d6f8d4b-a9bf-4c10-9ca2-50f2012c670e.png)<br>


# First import the required Python Libraries<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
from skimage import img_as_uint<br>
from skimage.io import imshow, imread<br>
from skimage.color import rgb2hsv<br>
from skimage.color import rgb2gray<br>
array_1 = np.array([[255, 255,0], <br>
                    [102,64, 0]])<br>
imshow(array_1);<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/181440461-296fc29d-2e9c-4e48-8b70-e49e0c1bae55.png)<br>


import numpy as np<br>
import matplotlib.pyplot as plt<br>
array_colors = np.array([[[245, 20, 36], <br>
                         [10, 215, 30],<br>
                         [40, 50, 205]],<br>
                         [[70, 50, 10], <br>
                    [25, 230, 85],<br>
                    [12, 128, 128]],<br>
                    [[25, 212, 3], <br>
                    [55, 5, 250],<br>
                    [240, 152, 25]],<br>
                    ])<br>
plt.imshow(array_colors)<br>
np.max(array_colors)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/181440768-50785dee-199b-46f5-90e4-3ce9daa83dac.png)<br>

<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
array_colors = np.array([[[245, 20, 36], <br>
                         [10, 215, 30],<br>
                         [40, 50, 205]],<br>
                         [[70, 50, 10], <br>
                    [25, 230, 85],<br>
                    [12, 128, 128]],<br>
                    [[25, 212, 3], <br>
                    [55, 5, 250],<br>
                    [240, 152, 25]],<br>
                    ])<br>
plt.imshow(array_colors)<br>
np.min(array_colors)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/181444731-48fc6286-8530-45c7-8d03-48d27f5a0375.png)<br>



import numpy as np<br>
import matplotlib.pyplot as plt<br>
array_colors = np.array([[[245, 20, 36], <br>
                         [10, 215, 30],<br>
                         [40, 50, 205]],<br>
                         [[70, 50, 10], <br>
                    [25, 230, 85],<br>
                    [12, 128, 128]],<br>
                    [[25, 212, 3], <br>
                    [55, 5, 250],<br>
                    [240, 152, 25]],<br>
                    ])<br>
plt.imshow(array_colors)<br>
np.std(array_colors)<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/181441053-35172757-3be8-4b39-af8f-9bbde3dfa627.png)<br>







