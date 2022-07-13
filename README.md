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


import cv2
OriginalImg=cv2.imread('flo1.jpg')
GrayImg=cv2.imread('flo1.jpg',0)
isSaved=cv2.imwrite('C:/i.jpg',GrayImg)
cv2.imshow('Display Original Image',OriginalImg)
cv2.imshow('Display Grayscale Image',GrayImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
if isSaved:
    print('The Image is successfully saved')
    OUTPUT:
    ![image](https://user-images.githubusercontent.com/97939356/178699172-4f8be175-c60b-42c3-913f-583ce7482dae.png)
    ![image](https://user-images.githubusercontent.com/97939356/178699314-dfe58cc5-989f-41de-8e8e-fc7da3d1cd37.png)
    
    
    import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('bf1.jpg',0)
x,y=image.shape
z=np.zeros((x,y))
for i in range(0,x):
    for j in range(0,y):
        if(image[i][j]>50 and image[i][j]<150):
            z[i][j]=255
        else:
            z[i][j]=image[i][j]
equ=np.hstack((image,z))
plt.title('Graylevel slicing with background')
plt.imshow(equ,'gray')
plt.show()
OUTPUT:
![image](https://user-images.githubusercontent.com/97939356/178703256-2f841cde-b8d8-43bd-946e-d81b8e4be626.png)

    











