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
8)Convert the original image to gray scale and then to binary 
import cv2
#read the image file
img=cv2.imread("flw22.jpg")
cv2.imshow("RGB",img)
cv2.waitKey(0)
#gray scale
img=cv2.imread("flw22.jpg",0)
cv2.imshow("gray",img)
cv2.waitKey(0)
#binary image
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("BINARY",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/97939356/174044748-26dfcc90-888b-4c63-abc3-853fd35d4ce4.png)
![image](https://user-images.githubusercontent.com/97939356/174045047-19658c38-e426-4b83-9d5a-1d2c2ef158c2.png)
![image](https://user-images.githubusercontent.com/97939356/174045302-fde2d346-0049-49a8-8da3-7c63501b68c8.png)




![image](https://user-images.githubusercontent.com/97939356/174043706-0a86ce89-0f01-4dcd-a402-43c03448d638.png)



