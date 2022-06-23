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
![image](https://user-images.githubusercontent.com/97939356/174043706-0a86ce89-0f01-4dcd-a402-43c03448d638.png)<br>
#develop a program to readimage using URL
from skimage import io
import matplotlib.pyplot as plt
url='https://images.hindustantimes.com/rf/image_size_630x354/HT/p2/2019/08/08/Pictures/_6bda0940-b9ad-11e9-98cb-e738ad509720.jpg'
image=io.imread(url)
plt.imshow(image)
plt.show()
output:
![image](https://user-images.githubusercontent.com/97939356/175019563-1ef4c35d-eefc-4fcd-b4d8-536d10c3351b.png)
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#reading image files
img1=cv2.imread('bf1.jpg')
img2=cv2.imread('bf3.jpg')
#applying numpy addition on images
fimg1=img1+img2
plt.imshow(fimg1)
plt.show()
#saving the outout image
cv2.imwrite('output.jpg',fimg1)
fimg2=img1-img2
plt.imshow(fimg2)
plt.show()
#saving the outout image
cv2.imwrite('output.jpg',fimg2)
fimg3=img1*img2
plt.imshow(fimg3)
plt.show()
#saving the outout image
cv2.imwrite('output.jpg',fimg3)
fimg4=img1/img2
plt.imshow(fimg4)
plt.show()
#saving the outout image
cv2.imwrite('output.jpg',fimg4)
1)![image](https://user-images.githubusercontent.com/97939356/175019733-b02ac822-3e74-4e0d-bf6c-cfe6fe70d17b.png)
2)![image](https://user-images.githubusercontent.com/97939356/175019810-a4738a4a-b4e4-4310-b99f-3c67090277b9.png)
3)![image](https://user-images.githubusercontent.com/97939356/175019905-57359335-99a4-4031-9f61-dc171bfc86dc.png)
4)![image](https://user-images.githubusercontent.com/97939356/175019986-237aa09f-a0dc-47cd-af68-94212b230495.png)

#write a program to mask and blur the image 
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mping.imread('bf55.jpg')
plt.imshow(img)
plt.show()
![image](https://user-images.githubusercontent.com/97939356/175020094-8dcc0acd-4d36-47f4-9ba7-6c7b75d40479.png)
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
light_orange=(1,190,200)
dark_orange=(18,255,255)
mask=cv2.inRange(hsv_img,light_orange,dark_orange)
result=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()
![image](https://user-images.githubusercontent.com/97939356/175020213-da03bde8-8c74-4ddb-976c-63f7ec657714.png)
light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()
![image](https://user-images.githubusercontent.com/97939356/175020311-4e0bf448-7b05-4807-9f1c-ea003ed934f3.png)
final_mask=mask+mask_white
final_result=cv2.bitwise_and(img,img,mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(final_mask)
plt.show()
![image](https://user-images.githubusercontent.com/97939356/175020431-3b05d13d-3fd0-4c05-b19d-74fe7b60f0c0.png)
blur=cv2.GaussianBlur(final_result,(7,7),0)
plt.imshow(blur)
plt.show()
![image](https://user-images.githubusercontent.com/97939356/175020540-fe6a5323-a391-4dbe-b705-0c8e9a38f1a3.png)

#develop the program to change the image to different color space
import cv2 
img=cv2.imread("bf1.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/97939356/175263981-f61e0c85-07d2-48e3-9ace-ba221ba6ff72.png)
![image](https://user-images.githubusercontent.com/97939356/175264136-a71009b9-93cf-45bf-a177-5b1a469e8658.png)
![image](https://user-images.githubusercontent.com/97939356/175264337-20846ea4-c8ea-4c9a-b5da-ce30f19ff8a0.png)
![image](https://user-images.githubusercontent.com/97939356/175264441-34c44e75-96e6-4132-99e7-70cb8f6148c1.png)
![image](https://user-images.githubusercontent.com/97939356/175265162-149969a5-fb00-4ab4-bbf9-df224b75cf99.png)












